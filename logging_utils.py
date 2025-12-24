import logging
import os


class _ContextFilter(logging.Filter):
    """Ensure optional logging fields exist to avoid KeyErrors."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        for key in ("node_id", "iteration", "phase"):
            if not hasattr(record, key):
                setattr(record, key, "-")
        return True


def _build_formatter() -> logging.Formatter:
    fmt = (
        "%(asctime)s [%(levelname)s] %(name)s "
        "| %(message)s"
    )
    return logging.Formatter(fmt)


def make_context_filter() -> logging.Filter:
    return _ContextFilter()


def make_formatter() -> logging.Formatter:
    return _build_formatter()


def init_logging(
    outdir: str,
    name: str,
    level: int = logging.INFO,
    to_console: bool = True,
) -> logging.Logger:
    """
    Initialize a module logger with a unified formatter and optional console/file outputs.

    Child loggers (e.g., ``logger.getChild("label")``) will inherit handlers and formatting
    without re-attaching handlers.
    """

    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = make_formatter()
    context_filter = make_context_filter()
    logger.addFilter(context_filter)

    if not logger.handlers:
        logfile = os.path.join(outdir, f"{name}.log")
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        fh.addFilter(context_filter)
        logger.addHandler(fh)

        if to_console:
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(formatter)
            sh.addFilter(context_filter)
            logger.addHandler(sh)

    logger.propagate = True
    return logger

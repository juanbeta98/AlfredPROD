import contextvars
import logging
from pathlib import Path

run_id_var = contextvars.ContextVar("run_id", default="-")
_record_factory = logging.getLogRecordFactory()


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        return True


def set_run_id(run_id: str) -> None:
    run_id_var.set(run_id)


def setup_logging_context() -> None:
    def record_factory(*args, **kwargs):
        record = _record_factory(*args, **kwargs)
        record.run_id = run_id_var.get()
        return record

    logging.setLogRecordFactory(record_factory)


def add_file_log_handler(log_path: Path) -> logging.FileHandler:
    """Attach a file handler to the root logger, writing to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | run_id=%(run_id)s | %(message)s"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(ContextFilter())
    logging.getLogger().addHandler(handler)
    return handler

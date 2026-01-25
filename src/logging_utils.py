import contextvars
import logging

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

"""Lightweight timing and memory diagnostics for pipeline operations."""

import logging
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


_active_context: ContextVar[str | None] = ContextVar(
    "sputum_colour_performance_context", default=None
)

logger = logging.getLogger("sputum_colour.performance")
logger.setLevel(logging.INFO)
logger.propagate = False
_file_handler: logging.FileHandler | None = None
_log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def configure_performance_logging(results_dir: str | Path) -> Path:
    """Write performance metrics to ``performance.log`` in the results directory."""
    global _file_handler

    log_path = Path(results_dir) / "performance.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        _file_handler is not None
        and Path(_file_handler.baseFilename) == log_path.resolve()
    ):
        return log_path

    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()

    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setFormatter(_log_formatter)
    logger.addHandler(_file_handler)
    return log_path


def _process_peak_rss_mb() -> float | None:
    if resource is None:
        return None

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak_rss == 0:
        return 0.0
    # macOS reports bytes; other supported Unix platforms report KiB.
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return peak_rss / divisor


def _format_memory(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f} MB"


@contextmanager
def measure_performance(operation: str) -> Iterator[None]:
    """Log elapsed time and memory measurements for a block of pipeline work."""
    root_scope = not tracemalloc.is_tracing()
    if root_scope:
        tracemalloc.start()
    python_start, _ = tracemalloc.get_traced_memory()
    rss_start = _process_peak_rss_mb()
    started_at = time.perf_counter()
    error_info = None

    try:
        yield
    except BaseException:
        error_info = sys.exc_info()
        raise
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        python_current, python_peak = tracemalloc.get_traced_memory()
        rss_end = _process_peak_rss_mb()

        fields = [
            f"operation={operation}",
            f"context={_active_context.get() or '-'}",
            f"elapsed={elapsed_seconds:.3f}s",
            f"status={'failed' if error_info else 'ok'}",
        ]
        if root_scope:
            fields.append(
                "python_traced_peak="
                f"{_format_memory((python_peak - python_start) / (1024**2))}"
            )
        else:
            fields.append(
                "python_traced_current_delta="
                f"{_format_memory((python_current - python_start) / (1024**2))}"
            )

        if rss_end is None:
            fields.append("process_peak_rss=unavailable")
        else:
            rss_growth = (
                max(0.0, rss_end - rss_start) if rss_start is not None else None
            )
            fields.append(f"process_peak_rss={_format_memory(rss_end)}")
            fields.append(f"process_peak_rss_growth={_format_memory(rss_growth)}")

        if root_scope:
            tracemalloc.stop()

        log_method = logger.error if error_info else logger.info
        log_method(" | ".join(fields), exc_info=error_info)


def monitored[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Measure a function without changing its arguments, return value, or errors."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        operation = f"{function.__module__}.{function.__qualname__}"
        with measure_performance(operation):
            return function(*args, **kwargs)

    return wrapper


def measure_each[T](items: Iterable[T], operation: str) -> Iterator[T]:
    """Measure each item while tagging nested operation logs with its identifier."""
    for item in items:
        context_token = _active_context.set(f"{operation}[{item}]")
        try:
            with measure_performance(f"{operation}[{item}]"):
                yield item
        finally:
            _active_context.reset(context_token)

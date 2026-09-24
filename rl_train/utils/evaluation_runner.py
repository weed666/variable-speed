import signal
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType
from typing import Callable

from rl_train.analyzer.train_analyzer import TrainAnalyzer


class EvaluationTimeoutError(TimeoutError):
    pass


@dataclass
class EvaluationRunResult:
    ok: bool
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None
    timed_out: bool = False


@contextmanager
def evaluation_timeout(timeout_seconds: float | None):
    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(signum: int, frame: FrameType | None) -> None:
        raise EvaluationTimeoutError(
            f"evaluation exceeded {timeout_seconds} seconds"
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_serial_evaluation(
    log_dir: str,
    *,
    timeout_seconds: float | None = None,
    analyzer_factory: Callable[[], TrainAnalyzer] = TrainAnalyzer,
) -> EvaluationRunResult:
    try:
        with evaluation_timeout(timeout_seconds):
            train_analyzer = analyzer_factory()
            train_analyzer.analyze_in_sequence(log_dir, show_plot=False)
    except EvaluationTimeoutError as exc:
        return EvaluationRunResult(
            ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
            timed_out=True,
        )
    except Exception as exc:
        return EvaluationRunResult(
            ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
        )

    return EvaluationRunResult(ok=True)

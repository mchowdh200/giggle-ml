import contextlib
import sys
import time
from contextlib import contextmanager

from giggleml.utils.torch_utils import get_rank


@contextmanager
def indent_prints(indent=2):
    """
    A context manager to indent all printed output within the block.
    """
    original_stdout = sys.stdout
    indent_string = " " * indent

    class IndentedOutput:
        def __init__(self, original_stream, indent_str):
            self.original_stream = original_stream
            self.indent_str = indent_str

        def write(self, s):
            # Only indent if the string is not empty and not just a newline
            if s and s != "\n":
                # Split by newlines, indent each line, then join back
                indented_s = "\n".join(
                    [self.indent_str + line for line in s.splitlines()]
                )
                # Write to the original stream, ensuring a newline for print()
                self.original_stream.write(indented_s)
            else:
                self.original_stream.write(s)  # Write empty or newline as is

        def flush(self):
            self.original_stream.flush()

    try:
        sys.stdout = IndentedOutput(original_stdout, indent_string)
        yield
    finally:
        sys.stdout = original_stdout  # Always restore original stdout


@contextmanager
def time_this(label: str = ""):
    """
    A context manager to time the execution of a code block.
    """
    start_time = time.perf_counter()

    try:
        if len(label) != 0:
            label = f"{label}: "
            print(f">>> {label}...")

        with indent_prints(3):
            yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"<<< {label}{elapsed_time:.4f} seconds")


def _format_time(seconds: float) -> str:
    """Converts a duration in seconds to a HH:MM:SS or MM:SS string."""
    if seconds < 0:
        return "0.0s"

    # Handle sub-minute durations
    if seconds < 60:
        return f"{seconds:.1f}s"

    # Handle durations less than an hour
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}m {remaining_seconds:02d}s"

    # Handle all other (longer) durations
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = int(seconds % 60)
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"


@contextlib.contextmanager
def progress_logger(
    total_steps: int, description: str, only_on_rank_zero: bool = False
):
    """
    A context manager for logging progress with a human-readable ETA.

    Yields a callback function `checkpoint()` that should be called
    at each step. Assumes steps are evenly weighted.
    """

    def no_op_checkpoint():
        pass

    if only_on_rank_zero and get_rank() != 0:
        try:
            yield no_op_checkpoint
        finally:
            return

    if total_steps <= 0:
        print(f"[{description}] Skipping (0 steps).")

        try:
            yield no_op_checkpoint
        finally:
            return

    start_time = time.monotonic()
    current_step = 0

    padding = " " * 10

    # Initial print
    print(
        f"\r[{description}] Starting... (0/{total_steps}) "
        f"| Elapsed: 0.0s | ETA: -{padding}",  # Added padding here too
        end="",
    )
    sys.stdout.flush()

    def checkpoint():
        """Updates and prints the progress."""
        nonlocal current_step
        current_step += 1

        elapsed_time = time.monotonic() - start_time

        # --- Calculations ---
        progress_fraction = min(1.0, current_step / total_steps)

        if progress_fraction > 0:
            time_per_step = elapsed_time / current_step
            remaining_steps = total_steps - current_step
            eta_seconds = max(0.0, time_per_step * remaining_steps)
        else:
            eta_seconds = 0.0

        # --- Formatting ---
        progress_percent = (current_step / total_steps) * 100
        elapsed_str = _format_time(elapsed_time)
        eta_str = _format_time(eta_seconds)

        # print the updated line
        print(
            f"\r[{description}] {progress_percent:5.1f}% ({current_step}/{total_steps}) "
            f"| Elapsed: {elapsed_str} | ETA: {eta_str}{padding}",
            end="",
        )
        sys.stdout.flush()

    try:
        yield checkpoint
    except Exception as e:
        # Clear the progress line on error
        print(f"\r{' ' * (len(description) + 70)}{padding}\r", end="")
        print(f"[{description}] Aborted with error: {e}")
        raise
    else:
        # Success: print final summary
        elapsed_time = time.monotonic() - start_time

        # Ensure final print is 100% and 0 ETA
        print(
            f"\r[{description}] {100.0:5.1f}% ({total_steps}/{total_steps}) "
            f"| Elapsed: {_format_time(elapsed_time)} | ETA: 0.0s{padding}",
            end="",
        )
        print(f"\n[{description}] Finished in {_format_time(elapsed_time)}.")

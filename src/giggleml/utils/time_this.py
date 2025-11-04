import contextlib
import sys
import time
from contextlib import contextmanager


# INFO: provided by gemini
@contextmanager
def indent_output(indent=2):
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

        with indent_output(3):
            yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"<<< {label}{elapsed_time:.4f} seconds")


@contextlib.contextmanager
def progress_logger(total_steps: int, description: str = "Processing"):
    """
    A context manager for logging progress with an ETA.

    Yields a callback function `checkpoint()` that should be called
    at each step. Assumes steps are evenly weighted.

    Args:
        total_steps (int): The total number of steps for the task.
        description (str): A description of the task to log.
    """

    # Handle edge case of no steps
    if total_steps <= 0:
        print(f"[{description}] Skipping (0 steps).")

        def no_op_checkpoint():
            pass

        try:
            yield no_op_checkpoint
        finally:
            return  # Exit without printing "Finished"

    start_time = time.monotonic()
    current_step = 0

    # Initial print
    print(f"\r[{description}] Starting... (0/{total_steps})", end="")
    sys.stdout.flush()

    def checkpoint():
        """Updates and prints the progress."""
        nonlocal current_step
        current_step += 1

        elapsed_time = time.monotonic() - start_time

        # --- Calculations ---
        progress_fraction = current_step / total_steps
        time_per_step = elapsed_time / current_step
        remaining_steps = total_steps - current_step
        eta_seconds = time_per_step * remaining_steps

        # --- Formatting ---
        progress_percent = progress_fraction * 100
        elapsed_str = f"{elapsed_time:.1f}s"

        # Handle ETA formatting
        if remaining_steps > 0:
            eta_str = f"{eta_seconds:.1f}s"
        else:
            # Don't show ETA if we're done or gone over
            eta_str = "0.0s"

        # Ensure progress doesn't exceed 100% if called exactly right
        if current_step == total_steps:
            progress_percent = 100.0

        # \r moves cursor to start of line.
        # Extra padding clears previous, longer lines.
        padding = " " * 10
        print(
            f"\r[{description}] {progress_percent:5.1f}% ({current_step}/{total_steps}) "
            f"| Elapsed: {elapsed_str} | ETA: {eta_str}{padding}",
            end="",
        )
        sys.stdout.flush()

    try:
        # Yield the callback for the 'with' block
        yield checkpoint
    except Exception as e:
        # Handle errors gracefully
        print(f"\n[{description}] Aborted with error: {e}")
        raise
    else:
        # Success case: Print final summary
        # This block runs if no exceptions were raised
        elapsed_time = time.monotonic() - start_time

        # Final update to show 100% if the loop finished
        # (This also neatly handles loops that finish faster than 0.1s)
        print(
            f"\r[{description}] {100.0:5.1f}% ({total_steps}/{total_steps}) "
            f"| Elapsed: {elapsed_time:.1f}s | ETA: 0.0s{padding}",
            end="",
        )
        print(f"\n[{description}] Finished in {elapsed_time:.2f}s.")

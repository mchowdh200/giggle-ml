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

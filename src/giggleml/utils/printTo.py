import sys
from contextlib import contextmanager


@contextmanager
def printTo(path: str):
    original_stdout = sys.stdout
    with open(path, "w") as f:
        sys.stdout = f
        yield
    sys.stdout = original_stdout

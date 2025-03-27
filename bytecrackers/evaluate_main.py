import contextlib
import io
import sys

from evaluation import evaluate
from your_implementation import implementation_main


def main():
    captured_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_output):
            implementation_main()
    except Exception as e:
        print(captured_output.getvalue())
        print("Your code raise an exception it will not ben evaluated.")
        print(e)
        raise e

    evaluate(captured_output, sys.argv[1])


if __name__ == "__main__":
    main()

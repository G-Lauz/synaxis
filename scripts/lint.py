"""
Git Hook that trigger on pre-commit. It run Pylint on all changed files within the
repository and execute the tests if the user is about to commit on the branch dev or
main.
"""

import argparse
import logging
import sys

from pylint import lint


def main():
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(prog="LINT")
    parser.add_argument(
        "-p",
        "--path",
        nargs="+",
        help="paths to directory to run Pylint.",
        default=["examples", "src", "tests"],
        type=str,
    )

    parser.add_argument(
        "-t",
        "--threshold",
        help="Score threshold to fail Pylint.",
        default=10.0,
        type=float,
    )

    args = parser.parse_args()
    paths = args.path
    threshold = args.threshold

    logging.info("Running Pylint within %s, will pass if the score is over: %s", paths, threshold)

    disabled = [
        "line-too-long",
        "missing-module-docstring",
    ]

    result = lint.Run(["--disable=" + ",".join(disabled), "--rcfile=pylintrc", *paths], exit=False)
    rating = result.linter.stats.global_note

    if rating < threshold and result.linter.stats.statement > 0:
        logging.error("Pylint failed, you must have a rating over %s/10.", threshold)
        sys.exit(1)
    else:
        logging.info("Pylint passed.")


if __name__ == "__main__":
    main()
    sys.exit(0)

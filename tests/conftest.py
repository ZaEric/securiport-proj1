import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-a",
        action="store",
        default=None,
        help="Path to the first run folder for reproducibility comparison.",
    )
    parser.addoption(
        "--run-b",
        action="store",
        default=None,
        help="Path to the second run folder for reproducibility comparison.",
    )
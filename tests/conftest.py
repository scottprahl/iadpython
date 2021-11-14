import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runbinary", action="store_true", default=False,
        help="run iad lib tests"
    )
    parser.addoption(
        "--notebooks", action="store_true", default=False, dest="notebooks",
        help="test notebooks by running them"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "iadc: mark test as needing libiad")
    config.addinivalue_line("markers", "notebooks: mark test as needing notebooks")


def pytest_collection_modifyitems(config, items):

    # Skip items marked with `iadc` by default
    if not config.getoption("--runbinary"):
        skip_binary = pytest.mark.skip(reason="need --runbinary option to run")
        for item in items:
            if "iadc" in item.keywords:
                item.add_marker(skip_binary)

    # Skip items marked with `notebooks` by default
    if not config.getoption("--notebooks"):
        skip_notebooks = pytest.mark.skip(reason="--notebooks option not used")
        for item in items:
            if "notebooks" in item.keywords:
                item.add_marker(skip_notebooks)

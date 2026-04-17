"""Console-script entry point for the extensionless `iadsum` script."""

from pathlib import Path
import runpy


def main():
    """Run the packaged `iadsum` script."""
    namespace = runpy.run_path(str(Path(__file__).with_name("iadsum")), run_name="__iadsum__")
    namespace["main"]()

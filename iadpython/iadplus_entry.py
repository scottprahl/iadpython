"""Console-script entry point for the extensionless `iadplus` script."""

from pathlib import Path
import runpy


def main():
    """Run the packaged `iadplus` script."""
    namespace = runpy.run_path(str(Path(__file__).with_name("iadplus")), run_name="__iadplus__")
    namespace["main"]()

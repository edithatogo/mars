"""Command line interface for pymars."""

from . import __version__


def main() -> None:
    """Simple CLI that prints the installed version."""
    print(f"pymars version {__version__}")


if __name__ == "__main__":
    main()

"""Command line interface for pymars."""

import logging
from . import __version__

logger = logging.getLogger(__name__)


def main() -> None:
    """Simple CLI that reports the installed version via logging."""
    logging.basicConfig()
    logger.info("pymars version %s", __version__)


if __name__ == "__main__":
    main()

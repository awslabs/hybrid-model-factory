"""
Custom logging module with colored output for hybrid model construction.
"""

import logging
import sys
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    # Log levels
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta

    # Layer types
    MAMBA2 = "\033[94m"  # Bright Blue
    BMOJO_F = "\033[95m"  # Bright Magenta
    GATED_DELTANET = "\033[96m"  # Bright Cyan
    GKA = "\033[35m"  # Magenta
    SWA = "\033[38;5;208m"  # Orange

    # Component types
    WEIGHTS = "\033[93m"  # Bright Yellow
    NORMS = "\033[92m"  # Bright Green
    SUMMARY = "\033[97m"  # Bright White

    # Special
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        for attr in dir(cls):
            if not attr.startswith("_") and attr.isupper() and attr != "disable":
                setattr(cls, attr, "")


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            levelname_color = (
                f"{self.LEVEL_COLORS[record.levelno]}"
                f"{levelname:8}"
                f"{Colors.RESET}"
            )
            record.levelname = levelname_color

        # Format the message
        result = super().format(record)
        return result


class HybridLogger:
    """Custom logger for hybrid model construction with colored output."""

    def __init__(self, name: str, level: int = logging.INFO, use_colors: bool = True):
        """
        Initialize the hybrid logger.
        
        Args:
            name: Logger name (typically __name__)
            level: Logging level
            use_colors: Whether to use colored output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        self.logger.propagate = False

        # Disable colors if requested or not in a terminal
        if not use_colors or not sys.stdout.isatty():
            Colors.disable()

        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = ColoredFormatter(
            fmt="%(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    # Layer-specific logging methods
    def mamba2_init(self, layer_idx: int, msg: str):
        """Log Mamba2 initialization."""
        colored_msg = f"{Colors.MAMBA2}[Mamba2 @ layer {layer_idx}]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def bmojo_f_init(self, layer_idx: int, msg: str):
        """Log B'MOJO initialization."""
        colored_msg = (
            f"{Colors.BMOJO_F}[B'MOJO @ layer {layer_idx}]{Colors.RESET} {msg}"
        )
        self.logger.info(colored_msg)

    def gated_deltanet_init(self, layer_idx: int, msg: str):
        """Log Gated DeltaNet initialization."""
        colored_msg = f"{Colors.GATED_DELTANET}[Gated DeltaNet @ layer {layer_idx}]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def gka_init(self, layer_idx: int, msg: str):
        """Log GKA initialization."""
        colored_msg = f"{Colors.GKA}[GKA @ layer {layer_idx}]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def swa_init(self, layer_idx: int, msg: str):
        """Log SWA initialization."""
        colored_msg = f"{Colors.SWA}[SWA @ layer {layer_idx}]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def weights_update(self, msg: str):
        """Log weight updates."""
        colored_msg = f"{Colors.WEIGHTS}[Weights]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def norms_update(self, layer_idx: int, msg: str):
        """Log norm layer updates."""
        colored_msg = f"{Colors.NORMS}[Norms @ layer {layer_idx}]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def summary(self, msg: str):
        """Log summary information."""
        colored_msg = f"{Colors.BOLD}{Colors.SUMMARY}[Summary]{Colors.RESET} {msg}"
        self.logger.info(colored_msg)

    def section(self, title: str):
        """Log a section header."""
        separator = "=" * 70
        colored_msg = (
            f"\n{Colors.BOLD}{Colors.SUMMARY}{separator}\n"
            f"{title}\n"
            f"{separator}{Colors.RESET}"
        )
        self.logger.info(colored_msg)


def get_logger(
    name: str, level: int = logging.INFO, use_colors: bool = True
) -> HybridLogger:
    """
    Get a configured hybrid logger.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        use_colors: Whether to use colored output (default: True)
        
    Returns:
        Configured HybridLogger instance
    """
    return HybridLogger(name, level, use_colors)

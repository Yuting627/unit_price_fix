"""Console + file logging with section helpers for check conclusions."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

FORMAT = "%(asctime)s %(levelname)s %(message)s"
ROOT_NAME = "peer"


def production_log_path(log_dir, period: int) -> Path:
    return Path(log_dir) / f"peer_outlier_{period}.log"


def get_logger(name: str = ROOT_NAME, log_file=None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to console and (optionally) ``log_file``.

    Handlers are replaced on each call so runs never share a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    fmt = logging.Formatter(FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def log_section(logger: logging.Logger, title: str, obj=None, max_rows: int = 30) -> None:
    """Log ``[TITLE]`` followed by a DataFrame / dict / message."""
    tag = f"[{title}]"
    if obj is None:
        logger.info(tag)
    elif isinstance(obj, pd.DataFrame):
        if obj.empty:
            logger.info("%s (empty)", tag)
        else:
            text = obj.head(max_rows).to_string(index=False)
            more = f"\n... {len(obj) - max_rows} more rows" if len(obj) > max_rows else ""
            logger.info("%s\n%s%s", tag, text, more)
    elif isinstance(obj, dict):
        logger.info("%s %s", tag, "; ".join(f"{k}={_fmt(v)}" for k, v in obj.items()))
    else:
        logger.info("%s %s", tag, obj)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

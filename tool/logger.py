"""全局彩色日志（检索侧节点使用）"""
import logging

import colorlog

_handler = colorlog.StreamHandler()
_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
))

logger = colorlog.getLogger("kb_query")
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

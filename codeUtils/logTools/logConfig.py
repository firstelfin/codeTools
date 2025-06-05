#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   logConfig.py
@Time    :   2024/08/18 19:20:54
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import sys
import logging
import logging.config as logging_config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path



LOG_COLORS={
    'DEBUG': 'cyan',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'white,bg red',
}
LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(process)d] - %(name)s - %(message)s"
LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_loguru(log_name="LOG/app.log", log_level="INFO", rotation="1 week", retention="30 days", file_handler=True, enqueue=False, **kwargs):
    """Setup loguru logging configuration.

    :param str log_name: log file name, defaults to "LOG/app.log"
    :param str log_level: log level, defaults to "INFO"
    :param str rotation: rotation time, defaults to "1 week"
    :param str retention: retention time, defaults to "30 days"
    :param bool file_handler: whether to use file handler, defaults to True
    :param bool enqueue: whether to use enqueue, defaults to False
    """
    from loguru import logger

    loguru_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | [<blue>{process}</blue>] - <yellow>{name}</yellow> - <level>{message}</level>"

    log_dir = Path(log_name).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format=loguru_format,
        colorize=True,
        level="INFO"
    )
    if file_handler:
        logger.add(
            log_name, rotation=rotation, retention=retention, 
            level=log_level.upper(), format=loguru_format, enqueue=enqueue
        )

    # 设置level颜色
    for level, level_color in LOG_COLORS.items():
        logger.level(level, color=f"<{level_color.replace(',', '><')}><bold>")


class MyFormatter(logging.Formatter):
    # ANSI escape sequences for bold style
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'GREEN': '\033[32m',
        'CYAN': '\033[36m',
        'RED': '\033[31m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'WHITE': '\033[37m',
        'BG RED': '\033[41m'
    }

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Original formatted message
        original_message = super().format(record)

        # 设置时间打印为绿色
        timestamp = self.formatTime(record, self.datefmt) + f".{int(record.msecs):03d}"
        colored_timestamp = f"{self.COLORS['GREEN']}{timestamp}{self.COLORS['RESET']}"
        # Apply bold style to levelname
        level_color = "".join([self.COLORS[c.upper()] for c in LOG_COLORS.get(record.levelname, 'WHITE').split(",")])
        bold_levelname = f"{self.COLORS['BOLD']}{level_color}{record.levelname}{self.COLORS['RESET']}"
        # 设置name为黄色
        yellow_name = f"{self.COLORS['YELLOW']}{record.name}{self.COLORS['RESET']}"
        # 设置process为蓝色
        blue_process = f"{self.COLORS['BLUE']}{record.process}{self.COLORS['RESET']}"
        # 设置message level_color
        color_msg = f"{self.COLORS['BOLD']}{level_color}{record.message}{self.COLORS['RESET']}"
        
        # Replace the levelname in the original message
        formatted_message = original_message.replace(record.levelname, bold_levelname)
        formatted_message = formatted_message.replace(timestamp, colored_timestamp)
        formatted_message = formatted_message.replace(record.name, yellow_name)
        formatted_message = formatted_message.replace(str(record.process), blue_process)
        formatted_message = formatted_message.replace(record.message, color_msg)

        return formatted_message


def setup_logger(log_name="LOG/app.log", log_level="INFO", backup_count=4, file_handler=True, **kwargs):
    """Setup logging configuration.

    :param str log_name: log file name, defaults to "LOG/app.log"
    :param str log_level: log level, defaults to "INFO"
    :param int backup_count: backup count for log file, defaults to 4
    :param bool file_handler: whether to use file handler, defaults to True
    """
    log_dir = Path(log_name).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # 更改fastapi、uvicorn的日志handler的格式

    config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "format": LOG_FORMAT,
                "datefmt": LOG_TIME_FORMAT
            },
            "access": {
                "format": LOG_FORMAT,
                "datefmt": LOG_TIME_FORMAT
            },
            "custom": {
                "()": MyFormatter,
                "format": LOG_FORMAT,
                "datefmt": LOG_TIME_FORMAT,
            },
            "file": {
                "format": LOG_FORMAT,
                "datefmt": LOG_TIME_FORMAT
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "custom",
                "level": "INFO",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "()": TimedRotatingFileHandler,  # 使用 TimedRotatingFileHandler
                "formatter": "file",  # 使用文件格式
                "level": log_level.upper(),
                "filename": log_name,  # 基础日志文件名
                "when": "W0",  # 每周一分割日志文件
                "interval": 1,  # 每周一个文件
                "backupCount": backup_count,  # 保留最近 backup_count 个文件
                "encoding": "utf-8"
            }
        },

        "loggers": {
            "uvicorn": {
                "handlers": ["console", "file"] if file_handler else ["console"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.error": {
                "handlers": ["console", "file"] if file_handler else ["console"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console", "file"] if file_handler else ["console"],
                "level": "INFO",
                "propagate": False
            }
        },
        "root": {
            "handlers": ["console", "file"] if file_handler else ["console"],
            "level": "INFO"
        }
    }

    # Apply logging configuration
    logging_config.dictConfig(config)  # 使用logging.config配置可能会因动态加载导致报错


def setup_log(**kwargs):
    """Setup logging configuration. kwargs: dict of log configuration.

    :param loguru: bool, whether to use loguru, default is False
    :type loguru: bool
    :param logging: bool, whether to use logging, default is True
    :type logging: bool
    :param log_name: str, log file name, default is "app.log"
    :type log_name: str
    :param log_level: str, log level, default is "INFO"
    :type log_level: str
    :param backup_count: int, backup count for log file, default is 4
    :type backup_count: int
    :param file_handler: bool, whether to use file handler, default is True
    :type file_handler: bool
    :param rotation: str, log rotation, default is "1 week"
    :type rotation: str
    :param retention: str, log retention, default is "30 days"
    :type retention: str
    :param enqueue: bool, whether to use enqueue, default is False
    :type enqueue: bool
    :return: None
    :rtype: NoneType

    Example:
    >>> setup_log(
    ...     loguru={
    ...         "log_name": "app.log", 
    ...         "log_level": "INFO", 
    ...         "rotation": "1 week", 
    ...         "retention": "30 days", 
    ...         "file_handler": True, 
    ...         "enqueue": False
    ...     }, 
    ...     logging={
    ...         "log_name": "app.log", 
    ...         "log_level": "INFO", 
    ...         "backup_count": 4, 
    ...         "file_handler": True
    ...     }
    ... )
    """
    if kwargs.get("loguru", None):
        setup_loguru(**kwargs["loguru"])
    if kwargs.get("logging", None):
        setup_logger(**kwargs["logging"])


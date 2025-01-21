import logging
from logging.config import dictConfig

LOG_LEVEL: str = ""

def init_logger(
        log_level : str ="INFO", 
        log_file : str ='log/log.txt'):
    """
    Initialize the logger with the specified log level and log file.
    
    Args:
        log_level (str): The log level (DEBUG, INFO, WARNING, ERROR).
        log_file (str): The path to the log file.
    """
    global LOG_LEVEL
    LOG_LEVEL = log_level
    LOG_FILE: str = log_file
    FORMAT: str = "\n%(asctime)s %(filename)s:%(lineno)d %(levelprefix)s\n\t%(message)s"

    class CustomFormatter(logging.Formatter):
        """
        Custom formatter
        """
        def format(self, record):
            level_prefix = {
                logging.DEBUG: "DEBUG",
                logging.INFO: "INFO",
                logging.WARNING: "WARNING",
                logging.ERROR: "ERROR",
            }.get(record.levelno, "UNKNOWN")
            record.levelprefix = f"{level_prefix}"
            return super().format(record)

    class CustomFormatterColor(logging.Formatter):
        """
        Custom formatter with color
        """
        def format(self, record):
            level_prefix = {
                logging.DEBUG: "\033[94mDEBUG\033[0m",
                logging.INFO: "\033[92mINFO\033[0m",
                logging.WARNING: "\033[93mWARNING\033[0m",
                logging.ERROR: "\033[91mERROR\033[0m",
            }.get(record.levelno, "UNKNOWN")
            record.levelprefix = f"{level_prefix}"
            return super().format(record)


    logging_config = {
        "version": 1,  # mandatory field
        "disable_existing_loggers": False,
        "formatters": {
            "custom": {
                "()": CustomFormatter,
                "fmt": FORMAT,
                "datefmt": "%H:%M:%S",
            },
            "custom_color": {
                "()": CustomFormatterColor,
                "fmt": FORMAT,
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "formatter": "custom_color",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": LOG_LEVEL,
            },
            "file": {
                "formatter": "custom",
                "class": "logging.FileHandler",
                "filename": LOG_FILE,
                "level": "DEBUG",
            }
        },
        "loggers": {
            "VIRAL": {
                "handlers": ["console", "file"],
                "level": LOG_LEVEL,
                # "propagate": False
            }
        },
    }
    dictConfig(logging_config)

def get_log_level():
    return LOG_LEVEL
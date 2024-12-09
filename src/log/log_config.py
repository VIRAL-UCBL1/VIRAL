from logging.config import dictConfig
import logging

def init_logger(
        log_level : str ="INFO", 
        log_file : str ='log/log.txt'):
    """Function that init the logger object that's properly setup"""
    
    LOG_FILE: str = log_file
    LOG_LEVEL: str = log_level
    FORMAT: str = "%(asctime)s %(filename)s:%(lineno)d %(levelprefix)s\n\t%(message)s\n"

    class CustomFormatter(logging.Formatter):
        """Custom formatter"""
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
        """Custom formatter"""
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
                "level": "INFO",
            }
        },
        "loggers": {
            "DREFUN": {
                "handlers": ["console", "file"],
                "level": LOG_LEVEL,
                # "propagate": False
            }
        },
    }
    dictConfig(logging_config)

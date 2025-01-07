import csv
import os
import re
from logging import getLogger

from Environments import EnvType


def getLoggerCSV():
    """Retrieve the instance of the CSV logger.

    Raises:
        NotImplementedError: If the logger instance has not been created.
    """
    if LoggerCSV._instance is None:
        raise NotImplementedError("need to instance the logger, before get it")
    return LoggerCSV._instance


class LoggerCSV:
    """Logger for recording performance metrics in a CSV file.

    Attributes:
        env_type (EnvType): The type of environment.
        llm (str): The language model used.
        csv_file (str): Path to the CSV file for logging.
        logger (Logger): Instance of the logger.
        _initialized (bool): Indicates whether the logger has been initialized.
    """
    _instance = None  # singleton

    def __new__(cls, env_type: EnvType, llm: str):
        """Create a new instance of LoggerCSV.

        Args:
            env_type (EnvType): The type of environment.
            llm (str): The language model used.

        Returns:
            LoggerCSV: The singleton instance of LoggerCSV.
        """
        if cls._instance is None:
            cls._instance = super(LoggerCSV, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_type: EnvType, llm: str):
        """Initialize the LoggerCSV instance.

        Args:
            env_type (EnvType): The type of environment.
            llm (str): The language model used.
        """
        if self._initialized:
            return
        self.env_type = env_type
        safe_env_type = re.sub(r'[^a-zA-Z0-9_]', '_', str(env_type))
        self.llm = llm
        self.csv_file = f"log/{safe_env_type}_log.csv"
        self.logger = getLogger("VIRAL")
        self._initialized = True

    def to_csv(self, state):
        """Write performance metrics to the CSV file.

        Args:
            state: The state object containing performance metrics.

        Raises:
            ValueError: If the state is not completed.
        """
        if state.performances is None:
            self.logger.debug(f"State {state.idx} is not completed")
            raise ValueError(f"State {state.idx} is not completed")
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w") as file:
                file.write("env;llm;reward_function;rewards;mean_reward;std_reward;SR\n")
        with open(self.csv_file, "a", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=";")
            spamwriter.writerow(
                [
                    self.env_type,
                    self.llm,
                    state.reward_func_str,
                    ','.join(map(str, state.performances["rewards"])),
                    state.performances["mean_reward"],
                    state.performances["std_reward"],
                    state.performances["sr"],
                ]
            )

import os
from Environments import EnvType
from logging import getLogger
import csv


def getLoggerCSV():
    if LoggerCSV._instance is None:
        raise NotImplementedError("need to instance the logger, before get it")
    return LoggerCSV._instance


class LoggerCSV:
    _instance = None  # singleton

    def __new__(cls, env_type: EnvType, llm: str):
        if cls._instance is None:
            cls._instance = super(LoggerCSV, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_type: EnvType, llm: str):
        if self._initialized:
            return
        self.env_type = env_type
        self.llm = llm
        self.csv_file = f"log/{env_type}_log.csv"
        self.logger = getLogger("VIRAL")
        self._initialized = True

    def to_csv(self, state):
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
                    state.performances["test_success_rate"],
                ]
            )

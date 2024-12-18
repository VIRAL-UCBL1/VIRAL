from Environments import EnvType
from logging import getLogger
import csv

def getLoggerCSV():
        if LoggerCSV._instance is None:
            raise NotImplementedError('need to instance the logger, before get it')
        return LoggerCSV._instance

class LoggerCSV:
    _instance = None #singleton

    def __new__(cls, env_type: EnvType, llm: str, csv_file: str = 'log/log.csv'):
        if cls._instance is None:
            cls._instance = super(LoggerCSV, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_type: EnvType, llm: str, csv_file: str = 'log/log.csv'):
        if self._initialized:
            return
        self.env_type = env_type
        self.llm = llm
        self.csv_file = csv_file
        self.logger = getLogger('VIRAL')
        self._initialized = True

    def to_csv(self, state):
        if state.performances is None:
            self.logger.debug(f'State {state.idx} is not completed')
            raise ValueError(f'State {state.idx} is not completed')
        with open(self.csv_file, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
            spamwriter.writerow([str(self.env_type), self.llm, state.reward_func_str, state.performances['test_success_rate']])

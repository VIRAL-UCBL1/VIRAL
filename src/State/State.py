from typing import Callable
from logging import getLogger
from log.LoggerCSV import getLoggerCSV

logger = getLogger("VIRAL")

class State:
    def __init__(
        self,
        idx,
        reward_func: Callable = None,
        reward_func_str: str = None,
        policy=None,
        perfomances: dict = None,
    ):
        """
        Represents a state in the reward function generation and evaluation process.

        This class encapsulates the key components of a reward function's lifecycle,
        tracking its index, implementation, policy, and performance metrics.

        Attributes:
            idx (int): Unique identifier for the state.
            reward_func (Callable, optional): The compiled reward function.
            reward_func_str (str, optional): String representation of the reward function.
            policy (object, optional): The policy associated with the reward function.
            performances (dict, optional): Performance metrics of the reward function.

        Key Characteristics:
            - Tracks the evolution of reward functions
            - Provides a snapshot of a specific iteration
            - Allows for dynamic updating of policy and performance

        Initialization Constraints:
            - Initial state (idx=0) cannot have a reward function
            - Non-initial states must have both a reward function and its string representation

        Methods:
            - set_policy(policy): Update the associated policy
            - set_performances(performances): Update performance metrics
            - __repr__(): Provide a human-readable string representation of the state

        Example:
            # Creating a new state for a reward function
            state = State(
                idx=1,
                reward_func=my_reward_func,
                reward_func_str="def reward_func(...):",
                policy=None,
                perfomances=None
            )

            # Updating state with training results
            state.set_policy(trained_policy)
            state.set_performances({
                'success_rate': 0.75,
                'average_reward': 10.5
            })

        Notes:
            - Designed for tracking reward function iterations
            - Provides flexibility in managing function states
            - Supports logging and debugging of reward function generation process
        """
        self.idx = idx
        self.src: list = [self.idx]
        if self.idx == 0 and (reward_func is not None or reward_func_str is not None):
            logger.error("the inital state don't take reward function")
        elif self.idx != 0 and (reward_func is None or reward_func_str is None):
            logger.error("you need to give a reward function to the state")
        self.reward_func = reward_func
        self.reward_func_str = reward_func_str
        self.policy = policy
        self.logger_csv = getLoggerCSV()
        self.performances = perfomances

    def set_src(self, state):
        self.src = state.src.copy()
        self.src.append(self.idx)

    def set_policy(self, policy):
        self.policy = policy

    def set_performances(self, performances: dict):
        self.performances = performances
        self.logger_csv.to_csv(self)

    def __repr__(self):
        if self.performances is None:
            repr = f"state {self.idx}: \nreward function: \n\n{self.reward_func_str}\n\n isn't trained yet"
        else:
            repr = f"state {self.idx}: \nreward function: \n\n{self.reward_func_str}\n\n success_rate: \n\n{self.performances['test_success_rate']}\n\n Policy: {self.policy}"
        return repr

from logging import getLogger
from typing import Callable
import numpy as np
from State.State import State
from LLM.OllamaChat import OllamaChat
from Environments import EnvType
from stable_baselines3.common.env_util import make_vec_env


class GenCode:
    def __init__(self, env: EnvType, llm: OllamaChat):
        """Generate the code from a response, it can be handle error, and refine from the llm new responses

        Args:
            env (Environments): description of the Environment
            llm (OllamaChat): the chat
        """
        self.current_index = 0
        self.llm = llm
        self.env_name = str(env)
        self.success_func = env.success_func
        self.logger = getLogger("VIRAL")
        self.response = None
        self.reward_func = None

    def get(self, response: str) -> State:
        """retrive a new State with reward_func

        Args:
            response (str): response code from the llm

        Returns:
            State: contain the Callable reward function and is string associeted
        """
        self.response = response
        self.current_index += 1
        self.reward_func = self.get_runnable_function()
        return State(self.current_index, self.reward_func, self.response)

    def get_runnable_function(self, error: str = None) -> Callable:
        """
        Process and validate a reward function for a gym environment.

        This method attempts to generate and validate a reward function by:\n
        1. Handling potential previous errors
        2. Creating a gym environment
        3. Cleaning and compiling the code
        4. Testing the reward function with a sample action
        5. Recursively handling various potential errors

        Args:
            response (str): The code response containing the reward function definition.
            error (str, optional): Previous error message to be added to LLM context.
                                    Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Callable: The compiled and validated reward function
                - str: The original response code

        Raises:
            - ValueError: Invalid function definition
            - SyntaxError: Syntax issues in the function
            - RuntimeError: Execution problems during function testing

        Note:
            - Uses recursion to handle potential errors
            - Relies on get_code, compile_reward_function, and test_reward_function methods
            - Provides a robust mechanism for generating valid reward functions
        """
        if error is not None:
            self.llm.add_message(error)
            self.response = self.llm.generate_response(stream=True)
            self.response = self.llm.print_Generator_and_return(self.response)
        try:
            env = make_vec_env(self.env_name)
            self.get_clean_response()
            reward_func = self.compile_reward_function()
            _ = env.reset()
            action = env.envs[0].action_space.sample()
            obs, _, _, infos = env.step([action])
            is_success = self.success_func(obs[0], infos[0])
            self.test_reward_function(
                reward_func, observations=obs[0], is_success=is_success
            )
        except ValueError as e:
            self.logger.warning(str(e))
            return self.get_runnable_function(str(e))
        except SyntaxError as e:
            self.logger.warning(f"Error syntax {e}")
            return self.get_runnable_function(str(e))
        except RuntimeError as e:
            self.logger.warning(f"Error execution {e}")
            return self.get_runnable_function(str(e))

        return reward_func

    def get_clean_response(self) -> None:
        """
        Clean and validate a code response by removing code block markers and ensuring a function definition.

        This method is designed to process code responses, typically extracted from text or code blocks,
        by performing the following operations:\n
        1. Remove leading and trailing code block markers (```),
        2. Remove the 'python' language identifier,
        3. Strip any additional whitespace
        4. Validate that the response contains a function definition

        Args:
            response (str): The raw code response to be cleaned and validated.

        Returns:
            str: The cleaned code response containing a function definition.

        Raises:
            ValueError: If the response does not contain a valid function definition
                        (i.e., if "def " is not present in the cleaned response).

        Logging:
            Logs the cleaned code at DEBUG level for debugging purposes.
        """
        start_idx = self.response.find("```")
        end_idx = self.response.find("```", start_idx + 1)
        if "def " not in self.response:
            raise ValueError("The answer does not contain a valid function definition.")
        if start_idx != -1 and end_idx != -1:
            cleaned_response = (
                self.response[start_idx:end_idx]
                .strip("```")
                .replace("python", "")
                .strip()
            )
        else:
            cleaned_response = self.response
        self.logger.debug("Code nettoyé pour compilation :\n" + cleaned_response)
        self.response = cleaned_response

    def compile_reward_function(self) -> Callable:
        """
        Compile a reward function dynamically from a string response.

        This method takes a code string representing a reward function and dynamically
        compiles it into an executable Python function. It provides a secure way to
        generate reward functions for reinforcement learning environments.

        Key Features:
            - Dynamically executes code in an isolated global namespace
            - Provides access to NumPy functions
            - Extracts the compiled function by its name
            - Robust error handling for syntax issues

        Args:
            response (str): A string containing a complete Python function definition
                            for a reward function.

        Returns:
            Callable: The compiled reward function that can be called with appropriate
                    arguments in a gym environment.

        Raises:
            SyntaxError: If the provided code contains invalid Python syntax.
            ValueError: If the function cannot be extracted from the compiled namespace.

        Notes:
            - Uses `exec()` for dynamic code compilation
            - Provides NumPy (`np`) in the execution namespace
            - Assumes the last function defined in the response is the reward function
        """

        exec_globals = {}
        exec_globals["np"] = np
        try:
            exec(self.response, exec_globals)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in the generated code : {e}")

        reward_function_name = self.response.split("(")[0].split()[
            -1
        ]  # récup le nom de la fonction
        reward_function = exec_globals.get(reward_function_name)

        return reward_function

    def test_reward_function(self, reward_function: Callable, *args, **kwargs):
        """
        Test the compiled reward function with provided inputs to validate its execution.

        This method serves as a crucial validation step in the reward function generation
        process. It attempts to execute the reward function with the given arguments and
        logs the output or raises an error if execution fails.

        Purpose:
            - Verify the reward function can be executed without errors
            - Log the reward function's output for debugging
            - Ensure the function returns a valid result in the context of a gym environment

        Args:
            reward_function (Callable): The compiled reward function to be tested.
            *args: Variable length argument list to pass to the reward function.
                Typically includes observations, actions, or environment states.
            **kwargs: Arbitrary keyword arguments to pass to the reward function.
                May include additional context like 'terminated' or 'truncated' flags.

        Raises:
            RuntimeError: If the reward function fails to execute successfully.
                This includes any exceptions that occur during function invocation.

        Logging:
            - Logs the reward function's output at DEBUG level when successful
            - Provides detailed error information if execution fails

        Notes:
            - Designed to be flexible with varying function signatures
            - Critical for validating dynamically generated reward functions
            - Part of the reward function generation quality control process
        """
        try:
            reward = reward_function(*args, **kwargs)
            self.logger.debug(f"Reward function output: {reward}")
        except Exception as e:
            raise RuntimeError(f"Error during reward function execution: {e}")

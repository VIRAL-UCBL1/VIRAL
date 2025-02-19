import base64
import json
from logging import getLogger
from typing import Dict, Generator, Optional, Union

import requests

OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"
OLLAMA_API_URL = "http://localhost:11434/api/generate"


class OllamaChat:
    """
    Class for an advanced Ollama chat session with extended configuration.
    """
    def __init__(
        self,
        model: str = "qwen2.5-coder",
        system_prompt: Optional[str] = None,
        options: Optional[Dict] = None,
        proxies: dict = None
    ) -> None:
        """
        Initialize an advanced Ollama chat session with extended configuration.

        Args:
            model (str, optional): The name of the Ollama model.
            system_prompt (str, optional): Initial system message to set chat context.
            options (dict, optional): Advanced model generation parameters.
            proxies (dict, optional): Proxy configuration for the HTTP requests.
        """
        self.proxies = proxies
        self.model = model
        self.messages: list[Dict[str, str]] = []
        self.options = options or {}
        self.logger = getLogger("VIRAL")

        if system_prompt:
            self.logger.info(f"System: {system_prompt}, Options: {self.options}")
            self.add_message(system_prompt, role="system")

    def add_message(self, content: str, role: str = "user", **kwargs) -> None:
        """
        Add a message to the chat history with optional metadata.

        Args:
            content (str): The message content
            role (str, optional): Message role (user/assistant/system)
            kwargs (dict): Additional message metadata
        """
        if 'images' in kwargs.keys():
            imgs_encoded = []
            for img_path in kwargs["images"]:
                with open(img_path, "rb") as image_file:
                    imgs_encoded.append(base64.b64encode(image_file.read()).decode('utf-8'))
            kwargs['images'] = imgs_encoded
        message = {"role": role, "content": content, **kwargs}
        self.messages.append(message)

    def generate_response(
        self, stream: bool = False, llm_options: Optional[Dict] = {}
    ) -> Union[str, Generator]:
        """
        Generate a response with advanced configuration options.

        Args:
            stream (bool, optional): Stream response in real-time
            llm_options (dict, optional): Temporary generation options

        Returns:
            Response as string or streaming generator
        """
        generation_options = {**self.options, **(llm_options or {})}

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": stream,
            "options": generation_options,
        }

        try:
            if self.proxies:
                response = requests.post(OLLAMA_CHAT_API_URL, json=payload, stream=stream, proxies=self.proxies)
            else:
                response = requests.post(OLLAMA_CHAT_API_URL, json=payload, stream=stream)
            response.raise_for_status()
            if not stream:
                full_response = response.json()
                assistant_response = full_response.get("message", {}).get("content", "")
                self.add_message(assistant_response, role="assistant")
                return assistant_response

            def stream_response():
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line.decode("utf-8"))
                            if "message" in json_response:
                                chunk = json_response["message"].get("content", "")
                                full_response += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue

                if full_response:
                    self.add_message(full_response, role="assistant")

            return stream_response()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error: {e}")
            return ""

    def generate_simple_response(
        self,
        prompt: str,
        sys_prompt: str = None,
        stream: bool = False,
        llm_options: Optional[Dict] = {},
    ):
        """
        Generate a simple response without historic.

        Args:
            prompt (str): user prompt
            sys_prompt (str, optional): system prompt
            stream (bool, optional): Stream response in real-time
            llm_options (dict, optional): Temporary generation options

        Returns:
            Response as string or streaming generator
        """
        generation_options = {**self.options, **(llm_options or {})}

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": sys_prompt,
            "stream": stream,
            "options": generation_options,
        }

        try:
            if self.proxies:
                response = requests.post(OLLAMA_CHAT_API_URL, json=payload, stream=stream, proxies=self.proxies)
            else:
                response = requests.post(OLLAMA_CHAT_API_URL, json=payload, stream=stream)
            response.raise_for_status()
            if not stream:
                full_response = response.json()
                assistant_response = full_response["response"]
                self.add_message(assistant_response, role="assistant")
                return assistant_response

            def stream_response():
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line.decode("utf-8"))
                            if "response" in json_response:
                                chunk = json_response["response"]
                                full_response += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue

                if full_response:
                    self.add_message(full_response, role="assistant")

            return stream_response()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection error: {e}")
            return ""

    def print_Generator_and_return(
        self, response: Generator | str, number: int = 1
    ) -> str:
        """
        Prints the content of a response if it is a generator, or simply returns the response as is.

        Args:
            response (Generator | str): The response to print or return. If it's a generator,
                                        it will be printed chunk by chunk. If it's a string,
                                        it will be returned directly.
            number (int, optional): The index of the response (default is 1). Used for logging purposes.

        Returns:
            The original response if it is a string, or the concatenated string of all chunks
            if it was a generator.
        """
        self.logger.info(f"Response {number}:")
        if isinstance(response, Generator):
            response_gen = response
            response = ""
            for chunk in response_gen:
                print(chunk, end="", flush=True)
                response += chunk
        return response


def main():
    chat = OllamaChat(
        model="qwen2.5-coder",
        system_prompt="""
        You are an expert in Reinforcement Learning specialized in designing reward functions. 
        Strict criteria:
        - Provide ONLY the reward function code
        - Use Python format
        - Briefly comment on the function's logic
        - Give no additional explanations
        - Focus on the Gymnasium Acrobot environment
        - STOP immediately after closing the ``` code block
        """,
        options={
            "temperature": 0.2,
            "max_tokens": 300,
        },
    )

    print("Programming conversation:\n")

    chat.add_message(
        "Implement a reward function for the Gymnasium Acrobot environment. I want only the reward_function() code with no additional explanations."
    )
    print(
        "User: Implement a reward function for the Gymnasium Acrobot environment. I want only the reward_function() code with no additional explanations."
    )

    print("\nAssistant: ", end="", flush=True)
    response = chat.generate_response(stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()

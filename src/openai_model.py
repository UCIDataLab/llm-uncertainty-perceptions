from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, 
)
from utils import parse_number

import openai, tiktoken, re
import numpy as np


def load_model(config_path: str, model_name: str, load_tokenizer=True, **kwargs) -> openai.OpenAI:    
    with open(config_path) as f:
        client = openai.OpenAI(api_key=f.read().strip(), **kwargs)
    
    if load_tokenizer:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        tokenizer = tiktoken.encoding_for_model(model_name).encode
    else:
        tokenizer = None
    return {
        "client": client,
        "model_name": model_name,
    }, tokenizer



@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(30))
def openai_chat(client, prompt: str, temperature: float, model_name: str, **kwargs):
    """OpenAI chat completion. Executes the specified prompt using the model and specified temperature."""
    def parse_response(resp):
        if kwargs.get("logprobs", False):
            return resp.logprobs.content[0].top_logprobs
        return resp.message.content
    
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        **kwargs
    )
    if kwargs.get("n", 1) == 1:
        return parse_response(response.choices[0])
    else:
        return [parse_response(choice) for choice in response.choices]

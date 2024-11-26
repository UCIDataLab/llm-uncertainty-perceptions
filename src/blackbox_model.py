from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict
from utils import parse_number

import numpy as np
import openai_model


def load_model(config_path: str, model_name: str, load_tokenizer=True, **kwargs):
    if "gemini" in model_name:
        import gemini_model as loader
    elif "together" in config_path:
        import together_model as loader
    else:
        import openai_model as loader
    return loader.load_model(config_path, model_name, load_tokenizer, **kwargs)


def get_chat_fn(client): 
    import google.generativeai as genai
    import openai as oai
    import together as tai
    
    if isinstance(client, genai.GenerativeModel):
        import gemini_model as gm
        return gm.gemini_chat
    elif isinstance(client, oai.OpenAI):
        import openai_model as om
        return om.openai_chat
    elif isinstance(client, tai.Together):
        import together_model as tm
        return tm.together_chat
    else:
        raise ValueError(f"Unsupported client: {type(client)}")


class CompletionBase(ABC):
    def __init__(self, client, model_name: str, max_new_tokens: int=1, temperature: float=1.0, parse_numbers: bool=True, **kwargs):
        self.client = client
        self.model_name = model_name
        self.model_kwargs = {k: v for k, v in kwargs.items()}
        self.model_kwargs.update({
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        })
        self.parse_numbers = parse_numbers
        self.fn = get_chat_fn(client)

    @abstractmethod
    def execute(self, prompt) -> Dict[str, list]:
        raise ValueError("This method must be implemented in a subclass.")


class SampleCompletion(CompletionBase):
    def __init__(self, sample_size: int, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(sample_size, int) and sample_size > 0, "The sample_size must be a positive integer."
        self.sample_size = sample_size

    def execute(self, prompt: str) -> Dict[str, list]:
        responses = defaultdict(list)
        for i in range(self.sample_size):
            response = self.fn(self.client, prompt=prompt, model_name=self.model_name, **self.model_kwargs)
            
            responses["model"].append(self.model_name)
            responses["model_kwargs"].append(self.model_kwargs)
            responses["model_input"].append(prompt)
            responses["response"].append(response)
            
            if self.parse_numbers:
                numbers = parse_number(response)
                has_number = len(numbers) > 0
                
                responses["response_has_number"].append(has_number)
                responses["response_first_number"].append(float(numbers[0]) if has_number else None)
                responses["response_last_number"].append(float(numbers[-1]) if has_number else None)
                responses["response_all_numbers"].append(numbers)
                responses["response_number_counts"].append(len(numbers))
                
                if not has_number:
                    print("="*120)
                    print(prompt)
                    print("-"*80)
                    print(response)
                    print("="*120)
        return responses
        

class LogProbsCompletion(CompletionBase):    
    def __init__(self, model_name, top_scores: int=20, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        assert top_scores > 0, f"Top scores must be a positive integer: {top_scores}"
        self.model_kwargs["logprobs"] = True
        self.model_kwargs["top_logprobs"] = top_scores
        self.top_scores = top_scores
        assert self.fn == openai_model.openai_chat, "LogProbsCompletion only supports OpenAI models"

    def execute(self, prompt: str) -> Dict[str, list]: 
        logprobs = self.fn(self.client, prompt=prompt, model_name=self.model_name, **self.model_kwargs)
        # As of 2024/04/15, the chat version only returns logprobabilities for immediate next step
        # To keep compatability with remainder of the interface, we will transform the output into
        # expected format, which is list of dicts with 20 keys <top token 1> ... <top token 5> and corresponding log probs 
        logprobs = [{lp.token: lp.logprob for lp in logprobs}]
        
        steps = []
        for step_scores in logprobs:
            # step scores is a dict with top_scores key value pairs
            top_tokens, top_scores = zip(*step_scores.items())
            top_metadata = {
                "model": self.model_name,
                "model_input": prompt,
                "top_prob_total": np.exp(top_scores).sum(),
                "top_prob_remaining": 1 - np.exp(top_scores).sum(),
                "top_tokens": top_tokens,
                "top_tokens_logprob": top_scores,
            }    
            if self.parse_numbers:
                number_count, number_prob = 1, 0
                has_number = False
                for token, score in zip(top_tokens, top_scores):
                    try:
                        num = float(token)
                        top_metadata[f"number_{number_count}"] = num
                        top_metadata[f"number_{number_count}_logprob"] = score
                        number_prob += np.exp(num)

                        number_count += 1
                        has_number = True
                    except:
                        pass
                for i in range(number_count, self.top_scores):
                    top_metadata[f"number_{i}"] = None
                    top_metadata[f"number_{i}_logprob"] = None
                top_metadata["response_has_number"] = has_number
                top_metadata["response_is_number"] = number_prob
            steps.append(top_metadata)
            
        if len(steps) == 0:
            print(logprobs)
            return {}
        
        # Put additional steps under a different kkey
        elif len(steps) > 1:
            steps[0]["extra_completions_steps"] = steps[1:]
        return {k: [v] for k, v in steps[0].items()}
        #^Note: ensures the output is dict[str, list]
        # (although a single example is returned, it's useful for compatibility)

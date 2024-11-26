from abc import ABC, abstractmethod
from collections import defaultdict
from scipy.special import logsumexp
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import parse_number

import torch
import numpy as np
import warnings


def get_model_size(canonic_name: str) -> int:
    import re 
    model_size = re.search(r"(\d+(\.\d+)?)(b|B|m|M)", canonic_name)
    if model_size is None:
        return -1
    val = model_size[0]
    const = 1e3 if val[-1] in ("b", "B") else 1        
    return float(val[:-1]) * const


def load_model(model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer from the given model name."""
    if "8x7" in model_name or get_model_size(model_name) > 12e9:
        device = "auto"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    model_kwargs = {"trust_remote_code": True} if "olmo" in model_name.lower() else   {}        
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


class CompletionBase(ABC):
    def __init__(self, model, tokenizer, max_new_tokens: int=1, temperature: float=1, parse_numbers: bool=True, **kwargs):
        self.model = model
        self.device = model.device
        self.model_name = model.config.name_or_path
        self.model_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature, 
            "do_sample": True,
        }
        
        if kwargs:
            print("Specified additional model kwargs:", kwargs)
            self.model_kwargs.update(kwargs)
            print("Current model_kwargs definitions:", self.model_kwargs)
        self.tokenizer = tokenizer
        self.parse_numbers = parse_numbers
        
    @abstractmethod
    def execute(self, prompt: str, **kwargs) -> Dict[str, list]:
        """Execute the completion for the given prompt."""
        pass

    @abstractmethod
    def collect_scores(self, prompt: str, completions: List[str], correct: bool=True) -> Dict[str, List[float]]:
        """Collect scores for the completions given the prompt."""
        pass


class SampleCompletion(CompletionBase):
    def __init__(self, sample_size: int, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(sample_size, int) and sample_size > 0, f"'sample_size' must be a positive integer: {sample_size}"
        self.sample_size = sample_size

    def execute(self, prompt: str, **kwargs) -> Dict[str, list]:
        if not kwargs:
            kwargs = self.model_kwargs
    
        responses = defaultdict(list)
        for i in range(self.sample_size):

            prompt_enc = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            prompt_enc = prompt_enc.to(self.device)
            
            # Generate the completions
            completions_ids = self.model.generate(prompt_enc, **kwargs)
            response = self.tokenizer.batch_decode(completions_ids[:, prompt_enc.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            if len(response) > 1:
                warnings.warn("More than one completion was generated. Only the first one will be considered.")
            response = response[0]
            
            responses["model"].append(self.model_name)
            responses["model_kwargs"].append(kwargs)
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
            return responses
    
    def collect_scores(self, prompt: str, completions: List[str], correct: bool=True) -> Dict[str, List[float]]:
        raise NotImplemented("Not implemented")


class LogProbsCompletion(CompletionBase):
    def __init__(self, top_scores: int=20, **kwargs):
        super().__init__(**kwargs)
        assert top_scores > 0, f"Top scores must be a positive integer: {top_scores}"
        self.top_scores = top_scores
        
        if self.parse_numbers:
            # Register the numerical tokens for a given model
            # (this is useful to correct probabilities later on)
            self.numerical_tokens: List[str] = sorted([key for key in self.tokenizer.vocab.keys() if key[0].isdigit() or key[0].isnumeric()])
            self.numerical_tokens_ids: List[int] = self.tokenizer.convert_tokens_to_ids(self.numerical_tokens)
        
    def execute(self, prompt: str) -> Dict[str, list]:
        """Sample completions for a given prompt up to ``max_new_tokens`` and a temperature of ``temperature``."""
        # Tokenize the prompt
        prompt_enc = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        prompt_enc = prompt_enc.to(self.device)
        # Generate the completions
        completions = self.model.generate(
            prompt_enc, return_dict_in_generate=True, output_scores=True, **self.model_kwargs)
        
        # Get top 5 information for every decoding step
        steps = []
        for step_scores in completions.scores:
            # re-normalize completion scores
            step_scores = torch.nn.functional.log_softmax(step_scores, dim=-1) # shape is 1 x vocab_size
            
            top_indices = torch.argsort(step_scores, descending=True, dim=-1)[0,:self.top_scores]
            top_scores = step_scores[0,top_indices].cpu().tolist()
            top_tokens = list(map(self.tokenizer.convert_ids_to_tokens, top_indices.cpu().tolist()))
            top_strings = list([[self.tokenizer.convert_tokens_to_string([t])] for t in top_tokens])
            top_metadata = {
                "model": self.model_name,
                "model_kwargs": self.model_kwargs,
                "model_input": prompt,
                "top_prob_total": np.exp(top_scores).sum(),
                "top_prob_remaining": 1 - np.exp(top_scores).sum(),
                "top_tokens": top_strings,
                "top_tokens_logprob": top_scores,
            }
            
            if self.parse_numbers:
                # Collect fine grained information about the `top_scores` tokens
                number_count = 1
                has_number = False
                number_prob = 0
                for token, score in zip(top_tokens, top_scores):
                    try:
                        num = float(token.strip())
                        top_metadata[f"number_{number_count}"] = num
                        top_metadata[f"number_{number_count}_logprob"] = score
                        has_number = True
                        number_count += 1
                        number_prob += np.exp(num)
                    except:
                        pass
                for i in range(number_count, self.top_scores):
                    top_metadata[f"number_{i}"] = None
                    top_metadata[f"number_{i}_logprob"] = None            
                top_metadata["response_has_number"] = has_number
                top_metadata["response_is_number"] = number_prob
            steps.append(top_metadata)

        # Put additional steps under a different kkey
        steps[0]["extra_completions_steps"] = steps[1:]            
        return {k: [v] for k, v in steps[0].items()}
        #^Note: ensures the output is dict[str, list]
        # (although a single example is returned, it's useful for compatibility)

    def collect_scores(self, prompt: str, completions: List[str], correct: bool=True) -> Dict[str, List[float]]:
        """Obtain scores for the corresponding completions conditioned on the prefix.
        This method executes |completions| + 1 model calls.
        
        Parameters
        ----------
        prompt: str
            The prompt (or prefix).
            
        completions: List[str]
            List of completions to append to the prompt and compute the scores for.
            
        correct: bool, defaults to True
            Whether to correct the log probabilities for numerical tokens. The corrected
            score will stored in column ``completion__cond_logscores__corrected``.
        
        Returns
        -------
        dict[str, list[float]]
            List of scores per completion, including jointscores, conditional scores, and
            prompt only scores.
        """
        def _get_cond_id(cond: str, nprompt: str) -> int:
            """Determines the index where cond and nprompt differ in the tokenization."""
            cond_ids = self.tokenizer(cond, add_special_tokens=True)["input_ids"]
            nprompt_ids = self.tokenizer(nprompt, add_special_tokens=True)["input_ids"]
            assert len(cond_ids) <= len(nprompt_ids)
            for i, (cid, npi) in enumerate(zip(cond_ids, nprompt_ids)):
                if cid != npi: 
                    return i
            return min(len(cond_ids), len(nprompt_ids))

        def _score_condprob(cond, nprompt, corrected: List[int]): 
            cond_id = _get_cond_id(cond, nprompt)
            nprompt_enc = self.tokenizer(nprompt, add_special_tokens=True, return_tensors="pt")
            nprompt_enc = nprompt_enc.to(self.device)
            
            with torch.no_grad():
                out = self.model(**nprompt_enc)

            logits = torch.log_softmax(out.logits.float(), dim=-1).detach()
            if corrected is not None and len(corrected) > 0: 
                select_not_num = torch.ones(logits.size(-1), dtype=torch.bool).to(self.model.device)
                select_not_num[corrected] = False
                corrected = logits[:, -1, select_not_num].to("cpu").numpy()
                corrected = logsumexp(corrected, axis=-1)

            #  The logits and inputs are mismatched by 1 
            logits = logits[:, :-1, :].to("cpu")
            input_ids = nprompt_enc.input_ids[:, 1:].to("cpu")
            # Create conditional mask
            cond_attn_mask = torch.zeros_like(nprompt_enc.input_ids.to("cpu"))
            cond_attn_mask[0, cond_id:] = 1
            cond_attn_mask = cond_attn_mask[:, 1:]
            
            gen_probs = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)        
            gen_probs2 = gen_probs * cond_attn_mask
            return gen_probs2.sum(dim=-1).cpu().tolist()[0], cond_attn_mask.sum(dim=-1).cpu().tolist()[0], corrected
        
        results = defaultdict(list)
        for completion in completions:
            nprompt = f"{prompt}{completion}"
            logprob, ntokens, logprob_nan = _score_condprob(prompt, nprompt, self.numerical_tokens_ids if self.parse_numbers and correct else None)
           
            results["completion__model"].append(self.model_name)
            results["completion__model_input"].append(nprompt)
            results["completion__cond"].append(prompt)
            results["completion__suffix"].append(completion)
            results["completion__cond_logscores"].append(logprob)
            results["completion__num_tokens"].append(ntokens)
            results["completion__cond_logscores_not_a_number_after"].append(logprob_nan)    

            if correct:
                assert len(logprob_nan) == 1
                logprob_nan = (logprob + logprob_nan).tolist()
                logprob_nan = logprob_nan[0]
                
                if logprob < logprob_nan:
                    _additional_details = f"\n-> log p('{completion}'|prompt) = {logprob}\n-> log p('{completion}', NaN|prompt) = {logprob_nan}"
                    warnings.warn(
                        f"Log probability difference should be >= 1e-9 but is {logprob - logprob_nan}: {_additional_details}",
                        RuntimeWarning,
                    )
            else:
                logprob_nan = logprob
            results["completion__cond_logscores__corrected"].append(logprob_nan)

        return results

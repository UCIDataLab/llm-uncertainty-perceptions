"""Non-verifiable script 

The functions of this script is to validate  the suitability of a prompt 
for producing numerical values between 0 and 100. Optionally, it also
uses probability-based approach (for whitebox methods) to compute the
probability of all integers in [0, 100]. 

For a given dataset and model proceed as follows:
1. Determine whether the maximum number of tokens required by the model to represent the digits from 0 to 100.
2. Determine whether the probability placed outside the digits is low enough to be considered negligible.
"""
from collections import defaultdict 
from functools import partial
from typing import Dict, List, Tuple

import prompts as pr
import hf_model as hf
import blackbox_model

import pandas as pd
import numpy as np
import argparse, logging, os, tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup_logger(path: str, logger=logger): 
    formatter = logging.Formatter('[%(asctime)s][%(name)s][--%(levelname)s--]: %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
    
    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(path, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Create a stream handler (output to terminal) and set the formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
        
    logger = logging.getLogger(__name__)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def is_path(text: str) -> bool:
    """Check if `text` resembles a path in the filesystem."""
    return isinstance(text, str) and any([text.startswith(char) for char in (".", "..", "/")]) or ":\\" in text


def get_filename(text: str):
    """Extract the filename from a path."""
    if is_path(text):
        if os.path.isdir(text):
            return text.rpartition("/")[-1]
        else:
            return text.split(".")[0].rpartition("/")[-1]
    else:
        return text


def parse_num_range(range: str, step=1) -> Tuple[int, int]:
    """Parse a numerical range expressed in a string.
    
    It assumes a range of integers, and the range is expressed 
    in the form of ``[lb,ub]``, ``lb,ub``, or (lb,ub).
    
    Returns
    -------
    Tuple[int, int]
        Lower bound and upper bound that can be plugged into a
        range.
    """
    lb, ub = range.split(",")
    if lb[0] == "[":
        lb = int(lb[1:])
    elif lb[0] == "(":
        lb = int(lb[1:]) + step
    else:
        lb = int(lb)
        
    if ub[-1] == "]":
        ub = int(ub[:-1])
    elif ub[-1] == ")":
        ub = int(ub[:-1]) - step
    else:
        ub = int(ub)
        
    return lb, ub
    
    
def max_number_of_tokens(encoder_fn, lower_bound: int, upper_bound: int) -> Dict[int, List[int]]:
    """Determines the maximum number of tokens needed to describe any 
    number between ``lower_bound`` and ``upper_bound``.
    
    Returns
    -------
    Dict[int, List[int]]
        The keys are the number of tokens needed to represent a number, 
        and the associated list is the list of numbers that require that
        many tokens.
    """
    results = defaultdict(list)
    for i in range(lower_bound, upper_bound+1):
        # Most models will represent the number with a whitespace, since they will provide a continuation to the sentence.        
        if (n := len(encoder_fn(f"{i}"))) > 1:
            results[n].append(i)
    return results

    
    
if __name__ == "__main__":
    # Example of usage with open source models (requires HuggingFace)
    # --------------------------------------------------------------
    # run_experiment.py \
    #   --model_name allenai/OLMo-7B-Instruct --device cuda \
    #   --input_path ./data/non-verifiable.csv \
    #   --prompt_file ./prompts/non_verifiable__2shot.json \
    #   --output_dir ./results/non-verifiable--2shot \
    #   --numerical_range \[0,100\]
    # 
    # Example of usage with OpenAI models:
    # --------------------------------------------------------------
    # run_experiment.py \
    #   --use_openai \
    #   --model_name gpt-4o-2024-05-13 \
    #   --input_path ./data/non-verifiable.csv \
    #   --prompt_file ./prompts/non_verifiable__2shot.json \
    #   --output_dir ./results/non-verifiable--2shot 
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--model_name", type=str, help="Name or local path of the model to load", required=True)
    parser.add_argument("--device", type=str, help="The device to load the model onto. Defaults to 'cuda' if available.", default="cuda")
    parser.add_argument("--input_path", type=str, help="Path to the input data. Expected format CSV", required=True)
    parser.add_argument("--prompt_file", type=str, help="The file with the prompt format to use for this model.", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to the output data", required=True)
    parser.add_argument("--numerical_range", type=str, help="The range of numerical values to validate. It assumes integer values.", default="[0,100]")
    parser.add_argument("--openai_config", type=str, help="OpenAI config file", default="./configs/openai.txt") # put OpenAI API key in a file
    parser.add_argument("--use_openai", action="store_true", help="Use the OpenAI model.")
    parser.add_argument("--use_sample", type=int, help="Use the self-consistency approach with 100 samples per statement.", default=None)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding during sampling.")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, get_filename(args.model_name))
    out_basename = "sample_completions" if args.use_sample is not None else "top5_completions"

    os.makedirs(output_dir, exist_ok=True)
    setup_logger(f"{output_dir}/{out_basename}.log", logger)
    logger.info(f"Experiment arguments:\n\n{args}\n\n")
    # Load the data
    input_data = pd.read_csv(args.input_path)
    logger.info(f"{len(input_data)} rows loaded successfully from {args.input_path}.")
    logger.debug(f"--------------- Example of data row ---------------")
    for k, v in input_data.iloc[0].items():
        logger.debug(f"{k}: '{v}'")
    # Load the model
    if args.use_openai:
        model, tokenizer = blackbox_model.load_model(args.openai_config, args.model_name)
        
        if args.use_sample is not None:
           completion = blackbox_model.SampleCompletion(sample_size=args.use_sample, **model, max_new_tokens=10, temperature=1)
        else:
            completion = blackbox_model.LogProbsCompletion(**model, max_new_tokens=1, temperature=1)

        encoder_fn = partial(tokenizer)
    else:
        model, tokenizer = hf.load_model(args.model_name, args.device)
        model.eval()
        encoder_fn = partial(tokenizer.encode, add_special_tokens=False)

        if args.use_sample is not None:
            if args.greedy:
                model_kwargs = {
                    "do_sample": False,
                    "num_beams": 1,
                }
            else:
                model_kwargs = {
                    "temperature": 1,
                }
            completion = hf.SampleCompletion(sample_size=args.use_sample, model=model, tokenizer=tokenizer, max_new_tokens=100, **model_kwargs)
        else:
            completion = hf.LogProbsCompletion(model=model, tokenizer=tokenizer, max_new_tokens=1, temperature=1)
            # ^Note: whenever the first token is a number, it means that there is definitely
            # a number being processed. The log probabilities or number collected should not
            # be trusted for all models since it is possible that the tokenizer represents
            # digits as individual tokens. However, it still provides useful signal as to
            # whether the model is likely to produce numbers.
            # ----------------------------------------------------------------------------

    lb, ub = parse_num_range(args.numerical_range)
    num_tokens = max_number_of_tokens(encoder_fn, lb, ub)
    
    # Report the number of tokens that is larger than 1 tokens
    max_tokens = max(num_tokens.keys()) if len(num_tokens) > 0 else 1
    for num_tokens, values in num_tokens.items():
        logger.debug(f"Fraction of examples with {num_tokens} tokens: {len(values)/(ub-lb+1):.2%}\n ->Values: {values}")

    prompt: pr.PromptBase = pr.PromptBase.from_path(args.prompt_file)
    logger.debug(f"Example prompt:\n{prompt.format(tokenizer, use_openai=args.use_openai)[0]}")
    
    if args.use_openai:
        completion_results, has_number = [], []
        for _, example in tqdm.tqdm(input_data.iterrows()):
            # get template in the right format
            template, template_placeholders = prompt.format(tokenizer, use_openai=args.use_openai)
            
            # replace placeholders
            for place_name, place_expr in template_placeholders.items():
                template = prompt.replace(template, place_expr, example[place_name])

            completions: Dict[str, list] = completion.execute(prompt=template)
            if len(completions) == 0:
                logger.warn(f"No completions for the example: {example}.\n\nIgnored example...")
                continue
            # Update information corresponding to that example
            n = len(completions["model"])
            completions.update({k: [v]*n for k, v in example.items()})
            
            completions = pd.DataFrame(completions)
            if len(completions) > 0:
                completion_results.append(completions)
            has_number.extend(completions["response_has_number"].tolist())
            
            if len(has_number) % (50 * (args.use_sample or 1)) == 0:
                logger.debug(f"After {len(has_number)} prompts: {np.mean(has_number):.2%} examples have at least one number.")

        if (mean := np.mean(has_number)) <= 0.50:
            logger.warning(f"Only {mean:.2%} of the prompts have at least one digit in the top completions.")

        # Save the results with top 5
        completion_results = pd.concat(completion_results, axis=0).reset_index(drop=True)
        completion_results.to_csv(os.path.join(output_dir, f"{out_basename}.csv"), index=False)
        logger.info(f"Results saved successfully for model: {args.model_name}.")
        
    else:
        logger.info("-"*80)
        logger.info(f"Compute the completions for all {ub-lb+1} numbers between {lb} and {ub}.")
        logger.info("-"*80)
        all_results = []

        if args.use_sample is not None:
            out_basename = "sample_completions.csv"
        else:
            out_basename = f"{lb}_to_{ub}_completions.csv"
        
        for ix, example in tqdm.tqdm(input_data.iterrows()):
            # get template in the right format
            template, template_placeholders = prompt.format(tokenizer, use_openai=args.use_openai)
        
            # replace placeholders
            for place_name, place_expr in template_placeholders.items():
                template = prompt.replace(template, place_expr, example[place_name])

            numbers_range = [f"{num}" for num in range(lb, ub+1)]
            if args.use_sample is not None:
                results = completion.execute(prompt=template)
            else:
                results = completion.collect_scores(prompt=template, completions=numbers_range, correct=True)
            
            n = len(next(iter(results.values())))
            results.update({k: [v]*n for k, v in example.items()})
            results = pd.DataFrame(results)
            all_results.append(results)
            
            if (len(all_results)+1) % 100 == 0:
                pd.concat(all_results).to_csv(os.path.join(output_dir, out_basename), index=False)
            
        all_results = pd.concat(all_results)
        all_results.to_csv(os.path.join(output_dir, out_basename), index=False)
        logger.info(f"All completions in {args.numerical_range} saved successfully for model: {args.model_name}.")
    logger.info("Process completed successfully.")

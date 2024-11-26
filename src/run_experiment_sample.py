"""Experiment utils

This script is a version of the `run_experiment` script modified to 
use sample decoding in the completion process. The motivation is to
be able to determine the models' abilities to perceive uncertainty
even in cases when logprobabilities are not easily available 
(eg, Gemini or Claude) or running the model is computationally
expensive (eg, LLAMA3 70B, DBRX).
"""
from typing import Dict

from run_experiment import (
    setup_logger,
    get_filename    
)

import prompts as pr
import blackbox_model

import pandas as pd
import numpy as np
import argparse, logging, os, tqdm, time


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

    
if __name__ == "__main__":
    # Example of usage with open source models through inference API
    # -------------------------------------------------------------------------
    # run_experiment_sample.py \
    #   --model_name meta-llama/Llama-3-70b-chat-hf \
    #   --input_path ./data/non-verifiable.csv \
    #   --prompt_file ./prompts/non_verifiable__2shot.json \
    #   --output_dir ./results/non-verifiable--2shot \
    #   --config_path ./configs/togetherai.txt
    # -------------------------------------------------------------------------
    # To run the decoding version with an open source model, you should use the other
    # script `run_experiment.py` instead. For instance, the following command will
    # use the preferred prompt to generate 1 sample for every row of the input data file.
    # run_experiment.py \
    #   --model_name allenai/OLMo-7B-Instruct \
    #   --device cuda 
    #   --input_path ./data/non-verifiable.csv \
    #   --prompt_file prompts/non_verifiable__2shot_preferred_prompts/allenai/OLMo-7B-Instruct.txt \
    #   --output_dir ./results/non-verifiable--2shot-preferred \
    #   --numerical_range \[0,100\] 
    #   --use_sample 1
    #   --greedy
    parser = argparse.ArgumentParser(description="Sample completions for a given blackbox model.")
    parser.add_argument("--model_name", type=str, help="Name or local path of the model to load.", required=True)
    parser.add_argument("--input_path", type=str, help="Path to the input data. Expected format CSV.", required=True)
    parser.add_argument("--prompt_file", type=str, help="The file with the prompt format to use for this model.", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to the output data.", required=True)
    parser.add_argument("--config_path", type=str, help="Path to a txt file with the API key.", required=True)
    parser.add_argument("--num_samples", type=int, help="Number of samples. Defaults to 1.", default=1)
    parser.add_argument("--temperature", type=int, help="Temperature to use when decoding.", default=0)
    parser.add_argument("--max_new_tokens", type=int, help="Max number of output tokens to generate.", default=200)
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Setup environment
    # -------------------------------------------------------------------------
    output_dir = os.path.join(args.output_dir, get_filename(args.model_name))
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(f"{output_dir}/sample_completions.log", logger)
    logger.info(f"Experiment arguments:\n\n{args}\n\n")
    
    # -------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------
    input_data = pd.read_csv(args.input_path)
    logger.info(f"{len(input_data)} rows loaded successfully from {args.input_path}.")
    logger.debug(f"--------------- Example of data row ---------------")
    for k, v in input_data.iloc[0].items():
        logger.debug(f"{k}: '{v}'")
        
    # -------------------------------------------------------------------------
    # Load the model
    # -------------------------------------------------------------------------
    model, tokenizer = blackbox_model.load_model(args.config_path, args.model_name, load_tokenizer=False)
    completion = blackbox_model.SampleCompletion(sample_size=args.num_samples, **model, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    prompt: pr.PromptBase = pr.PromptBase.from_path(args.prompt_file)
    logger.debug(f"Example prompt:\n{prompt.format(tokenizer, use_openai=True)[0]}") 
    # ^Note: we hard code use_openai=True because we're mostly running this code
    # using black box API that resemble OpenAI API and we wish to preserve the
    # format of the original prompt file (no need to apply additional processing steps)
    
    completion_results, has_number = [], []
    for _, example in tqdm.tqdm(input_data.iterrows()):
        # get template in the right format
        template, template_placeholders = prompt.format(tokenizer, use_openai=True)
        
        # replace placeholders
        for place_name, place_expr in template_placeholders.items():
            template = prompt.replace(template, place_expr, example[place_name])
        
        try:
            completions: Dict[str, list] = completion.execute(prompt=template)            
        except: 
            time.sleep(180)
            try:
                completions: Dict[str, list] = completion.execute(prompt=template)
            except:
                completions = []
            
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
        
        if (len(has_number) + 1) % (100 * (args.num_samples)) == 0:
            logger.debug(f"After {len(has_number)} prompts: {np.mean(has_number):.2%} examples have at least one number.")
            results = pd.concat(completion_results, axis=0).reset_index(drop=True)
            results.to_csv(f"{output_dir}/sample_completions.csv", index=False)

        results = pd.concat(completion_results, axis=0).reset_index(drop=True)
        results.to_csv(f"{output_dir}/sample_completions.csv", index=False)
        logger.info(f"Results saved successfully for model: {args.model_name}.")
    logger.info("Process completed successfully.")

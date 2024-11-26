from collections import defaultdict

from run_experiment1 import get_filename
from utils import parse_number

import pandas as pd 
import argparse, os, tqdm

import prompts as pr
import openai_model as oai


def setup_env(args):
    # create output directory
    model_name = get_filename(args.model_name)
    model_name = model_name.replace("/", "__")
    
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_filepath = f"{args.output_dir}/{model_name}.csv"


def load_client(args):
    kwargs = dict(config_path=args.config_filepath, model_name=args.model_name, load_tokenizer=False)
    if args.base_url is not None:
        kwargs["base_url"] = args.base_url
    return oai.load_model(**kwargs)


def execute(model, tokenizer, prompt, input_data, n_samples, temperature, max_tokens) -> pd.DataFrame:
    results = defaultdict(list)
    for _, example in tqdm.tqdm(input_data.iterrows()):
        # get template in the right format
        template, template_placeholders = prompt.format(tokenizer, use_openai=True)
        
        # replace placeholders
        for place_name, place_expr in template_placeholders.items():
            template = prompt.replace(template, place_expr, example[place_name])

        for i in range(n_samples):
            response = oai.openai_chat(**model, temperature=temperature, max_tokens=max_tokens, prompt=template)
            numbers = parse_number(response)
            has_number = len(numbers) > 0
            
            for k, v in example.items():
                results[k].append(v)
            
            results["input"].append(template)
            results["response"].append(response)
            results["has_number"].append(has_number)
            
            if has_number:
                number_idx = response.index(numbers[0])
                results["response_prefix"].append(response[:number_idx])
            else:
                results["response_prefix"].append(None)
            
    return pd.DataFrame(results)
    
    
if __name__ == "__main__":
    # python src/run_prompt.py \
    #   --config_filepath ~/configs/togetherai.txt \
    #   --base_url https://api.together.xyz/v1 \
    #   --model_name google/gemma-2b-it \
    #   --prompt_filepath ./prompts/non_verifiable__2shot.json \
    #   --input_filepath ./data/non-verifiable.csv \
    #   --output_dir ./results/run_prompt/non-verifiable --n 1
    parser = argparse.ArgumentParser(
        description='Useful script to run quick tests to discern a model\'s adherence to a prompt.')
    parser.add_argument('--config_filepath', type=str, help='API Key', required=True)
    parser.add_argument('--base_url', type=str, help='Base url', default=None)

    parser.add_argument('--model_name', type=str, help='Model name', required=True)
    parser.add_argument('--prompt_filepath', type=str, help='Prompt', required=True)
    parser.add_argument('--input_filepath', type=str, help='Input filepath', required=True)
    parser.add_argument('--output_dir', type=str, help='Output', required=True)
    parser.add_argument('--n', type=int, help='Number of samples', default=5)

    args = parser.parse_args()
    
    setup_env(args)
    # Load model
    model, tokenizer = load_client(args)
    
    # Load prompt 
    prompt: pr.PromptBase = pr.PromptBase.from_path(args.prompt_filepath)
    print(f"Example prompt:\n{prompt.format(tokenizer, use_openai=True)[0]}")

    # Load dataset 
    input_data = pd.read_csv(args.input_filepath).sample(frac=0.5, random_state=91823, replace=False)

    # Execute
    results = execute(model, tokenizer, prompt, input_data, n_samples=args.n, temperature=1, max_tokens=100)
    results.to_csv(args.output_filepath, index=None)

# Run prompt analysis

- [x] Run OLMo instruct. 
- Findings: Out of 450 examples 
    - ~80% examples produce a number prefixed with " ".
    - ~12% examples repeat the instructions "in this experiment, you will be shown..."
    - ~5% examples the model refuse to answer.

```bash
$ python src/run_prompt.py \
    --model_name allenai/OLMo-7B-Instruct
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] Run LLAMA 3 70B Chat in together AI (I'm guessing meta-llama/Meta-Llama-3-70B-Instruct in HF)
- Findings: Out of 450 examples
    - 100% have produced numbers.
    - 303 examples exhibit the " " before the number.
    - 19 examples prefix the answer with "**"
    - 16 examples prefix the answer with some variation of "I would answer", "Based on the sentence, I would answer", "The correct answer is" and "\n\n".
    - The remaining examples contain explanations.
    
```bash
$ python src/run_prompt.py \
    --model_name meta-llama/Llama-3-70b-chat-hf \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] Run Mistral 7B Instruct  `mistralai/Mistral-7B-Instruct-v0.2`
- Findings: 
    - 32 examples do not produce numbers and instead refuses to answer due to the lack of information about the speaker's belief.
    - 418 examples produce numbers, of these 298 are prefixed with " " and another 30 examples correspond to variations of "The correct answer is"

```bash
$ python src/run_prompt.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] Run Mistral 8X7B Instruct  `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Findings: Of the 450 examples:
    - 447 examples are prefixed with " ".

```bash
$ python src/run_prompt.py \
    --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] Run Gemma 7B Instruct analysis 
- Findings: of 450 examples
    - 286 did not produce a number and instead refuses to answer (x% of the time) or 
    - 164 examples did produce a number but most of them concerns the paraphrasing or regurgitation of the experiment instructions.

For this reason, we will exclude Gemma 7B Instruct from our analysis.

```bash
$ python src/run_prompt.py \
    --model_name google/gemma-7b-it	 \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```
- [x] Run Gemma 2B Instruct analysis 
- Findings: of 450 examples
    - 447 produced a number and 3 didn't.
    - From which only 2 of them are prefixed "**" and the remaining models are prefixed with "".

```bash
$ python src/run_prompt.py \
    --model_name google/gemma-2b-it	 \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] Run DBRX Instruct analysis 
- Findings: Of the 450 examples
    - 450 examples produce number as a response to the prompt using prefix "". Note that they do not add a whitespace.
```bash
$ python src/run_prompt.py \
    --model_name databricks/dbrx-instruct \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [ ] Run Open Chat  `openchat/openchat-3.5-1210`

```bash
$ python src/run_prompt.py \
    --model_name openchat/openchat-3.5-1210 \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] Run lmsys/vicuna-13b-v1.5	
- Findings: Of 450 examples tested:
    - 438 have a number (with 45 unique prefixes)
        - 379 start with " "
        - 18 start with some variation of "The correct answer is", "Correct Answer:", "Answer:". 
        - 20 start with some variation of "Based on the given sentence [...] answer would be" or "Based on the given sentence [...] answer is between"
    - 12 don't have a number
        - 7 of which the model refuses to aswer, stating that it doesn't have enough information and asks for additional information.
        
```bash
$ python src/run_prompt.py \
    --model_name lmsys/vicuna-13b-v1.5	 \
    --config_filepath ~./configs/togetherai.txt \
    --base_url https://api.together.xyz/v1 \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```


- [x] ChatGPT 
- Findings: Of 450 examples tested:
    - 450 produce a number preceeded by no whitespace and comprising 21 different numbers.

```bash
$ python src/run_prompt.py \
    --model_name gpt-3.5-turbo-0125	 \
    --config_filepath ~./configs/openai.txt \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```

- [x] GPT4 
- Findings: Of 450 examples tested:
    - 449 produce number, 447 of which produce a number with no additional prefix. The two prefixes are produced to the control expression using no uncertainty expression.
    - Though it cautions the user that there is not enough information (again only for the control expression), it still produces a number.
    
```bash
$ python src/run_prompt.py \
    --model_name gpt-4-turbo-2024-04-09	 \
    --config_filepath ../configs/openai.txt \
    --prompt_filepath ./prompts/non_verifiable__2shot.json \
    --input_filepath ./data/non-verifiable.csv \
    --output_dir ./prompt_experiments/ \
    --n 1
```
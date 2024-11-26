from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import together


def load_model(config_path: str, model_name: str, load_tokenizer=True, **kwargs):    
    with open(config_path) as f:
        client = together.Together(api_key=f.read().strip(), **kwargs)
    
    if load_tokenizer:
        # not sure how to get token 
        tokenizer = lambda _: 0
    else:
        tokenizer = None
    return {
        "client": client,
        "model_name": model_name,
    }, tokenizer
    
    

@retry(wait=wait_random_exponential(min=1, max=6), stop=stop_after_attempt(30))
def together_chat(client, prompt: str, temperature: float, model_name: str, **kwargs):
    """Together AI chat completion.
    
    Executes the specified prompt using the specified model and parameters."""
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
    if kwargs.get("logprobs", False):
        raise NotImplementedError("Not implemented...")
    return response.choices[0].message.content

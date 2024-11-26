from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, 
)
import google.generativeai as genai
import warnings


def load_model(config_path: str, model_name: str, load_tokenizer=True, **kwargs):    
    with open(config_path) as f:
        genai.configure(api_key=f.read().strip(), **kwargs)
    
    # create model
    model = genai.GenerativeModel(model_name)
    
    if not load_tokenizer:
        tokenizer = model.count_tokens
    else:
        tokenizer = None
    
    return {
        "client": model,
        "model_name": model_name,
    }, tokenizer



@retry(wait=wait_random_exponential(min=1, max=6), stop=stop_after_attempt(30))
def gemini_chat(client: genai.GenerativeModel, prompt: str, temperature: float, max_tokens: int, candidate_counts=1, **kwargs):
    """Google generative ai chat completion. Executes the specified prompt using the model and specified temperature."""
    # https://cloud.google.com/vertex-ai/generative-ai/docs/chat/test-chat-prompts#chat-query-python_vertex_ai_sdk
  
    if isinstance(prompt, str):
        messages = prompt
    else:
        if len(prompt) > 1:
            warnings.warn(f"Chat prompt is currently not supported by Gemini chat completion: {prompt}")
        messages = prompt[0].get("content", "")

    responses = []
    for _ in range(candidate_counts):
        parameters = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 1,
        }
    
        response = client.generate_content(messages, generation_config=parameters)
        responses.append(response.parts[0].text)
    return responses[0] if candidate_counts == 1 else responses
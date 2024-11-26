from abc import ABC, abstractmethod
from typing import Dict, List, Union

import copy, json, re, yaml


class PromptBase(ABC):
    def from_path(path: str, **kwargs):
        with open(path, "r") as f:
            if path.endswith(".txt"):
                return PromptForBase(f.read())
            elif path.endswith(".json"):
                return PromptForChat(json.load(f))
            elif path.endswith(".yml") or path.endswith(".yaml"):
                return PromptForChat(yaml.load(f))
            else:
                raise ValueError(f"File format not supported: {path}")
        
    @classmethod
    def get_placeholders(cls, text: str) -> Dict[str, str]:
        """For every text returns the placeholder field corresponding to the placeholder expression.
        
        For example, executing the placeholders method in the text:
        "When prompted to provide a number, do [[something]].", will return:
        {"something": "[[something]]"}
         
        The assumption is that the user is insterested in using the placeholder
        name to fetch information and not the placeholder itself. However, if the
        user prefers using str.replace to replace the placeholder with a value, the
        structure also provides that possibility.
        """
        plac = re.findall(r"\[\[(.+)?\]\]", text)
        return {t: f"[[{t}]]" for t in plac}
        
    @abstractmethod
    def format(self, tokenizer, use_openai: bool):
        raise ValueError("This method must be implemented in a subclass.")


class PromptForBase(PromptBase):
    def __init__(self, text: str, **kwargs):
        self.message = text        
        
    def placeholders(self) -> Dict[str, str]:
        return PromptBase.get_placeholders(self.message)
    
    def format(self, tokenizer, use_openai: bool):
        return self.message, self.placeholders()
    
    def replace(self, messages, placeholder, placeholder_value):
        return messages.replace(placeholder, placeholder_value)


class PromptForChat(PromptForBase):
    def __init__(self, texts: List[Dict[str, str]], text_col: str="content", **kwargs):
        assert isinstance(texts, list) and isinstance(texts[0], dict), "The texts must be a list of dictionaries."
            
        self.messages = texts
        self.content_col = text_col

    def placeholders(self) -> Dict[str, str]:
        pls = {}
        for message in self.messages:
            pls.update(PromptBase.get_placeholders(message[self.content_col]))
        return pls

    def format(self, tokenizer, use_openai: bool):
        if use_openai:
            return self.messages, self.placeholders()
        else:
            try:
                return tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True), self.placeholders()
            except:
                system_msg = [m[self.content_col] for m in self.messages if m["role"].lower() == "system"][0] 
                other_msgs = [m for m in self.messages if m["role"].lower() != "system"]
                
                other_msgs[0][self.content_col] = system_msg + "\n\n" + other_msgs[0][self.content_col] # add system message to first message
                self.messages = other_msgs
                return tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True), self.placeholders()

    def replace(self, messages, placeholder, placeholder_value):
        if isinstance(messages, str):
            return messages.replace(placeholder, placeholder_value)
        
        # open ai models will preserve messages
        updated_msgs = []
        for msg in copy.deepcopy(messages):
            msg[self.content_col] = msg[self.content_col].replace(placeholder, placeholder_value)
            updated_msgs.append(msg)
        return updated_msgs
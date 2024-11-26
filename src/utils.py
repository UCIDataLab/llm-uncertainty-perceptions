from typing import List

import re 


def parse_number(text: str, regex=r"(\b\d+\b)") -> List[str]:
        return re.findall(regex, text)
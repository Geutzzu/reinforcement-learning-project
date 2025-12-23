
from typing import List
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

qwen_tokenizer = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "tokenizer.json"))

def batch_tokenize(texts: List[str], batch_size: int = 1000) -> List[List[int]]:
    token_lens = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = qwen_tokenizer.encode_batch(batch)
        token_lens.extend(len(enc.ids) for enc in encodings)
    return token_lens
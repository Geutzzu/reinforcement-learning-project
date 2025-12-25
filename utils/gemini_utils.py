from google import genai
from typing import List, Optional

API_KEY = open("../gemini_key.txt").read().strip()

class GeminiLLM:
    def __init__(self, model: str = "gemini-3-flash-preview", **kwargs):
        self.model_name = model
        self.client = genai.Client(api_key=API_KEY)
    
    def predict_batch(self, prompts: List[str], max_output_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20, stop: Optional[List[str]] = None, **kwargs) -> List[str]:
        results = []
        
        config = genai.types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop or [],
        )
        
        for prompt in prompts:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            results.append(response.text)
        
        return results
"""
python merge_adapter.py --adapter-path /path/to/adapter --output-path /path/to/merged
python merge_adapter.py --adapter-path /path/to/adapter 
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def merge_adapter(adapter_path: str, output_path: str = None):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    model = model.merge_and_unload()
    
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter-path", type=str, required=True, help="Absolute path to the trained adapter (checkpoint directory)")
    parser.add_argument("--output-path", type=str, default=None, help="Absolute path to save the merged model. Defaults to 'merged' next to adapter")
    
    args = parser.parse_args()
    merge_adapter(args.adapter_path, args.output_path)


if __name__ == "__main__":
    main()

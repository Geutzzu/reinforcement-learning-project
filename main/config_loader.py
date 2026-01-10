import yaml
from dataclasses import fields, asdict
from pathlib import Path
from config import BaseConfig, SFTConfig, GRPOConfig, RLOOConfig

CONFIG_CLASSES = {"sft": SFTConfig, "grpo": GRPOConfig, "rloo": RLOOConfig}


def load_config(path: str, config_type: str = "sft") -> BaseConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    
    cls = CONFIG_CLASSES[config_type]
    
    valid_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def save_config(config: BaseConfig, path: str):
    data = asdict(config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_default_configs(output_dir: str = "configs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_config(SFTConfig(), f"{output_dir}/sft_default.yaml")
    save_config(GRPOConfig(), f"{output_dir}/grpo_default.yaml")
    print(f"Generated default configs in {output_dir}/")


if __name__ == "__main__":
    generate_default_configs()

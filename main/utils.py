from datasets import load_dataset


def load_datasets(config):
    train = load_dataset("parquet", data_files=config.train_dataset_path, split="train")
    
    if config.eval_dataset_path:
        eval = load_dataset("parquet", data_files=config.eval_dataset_path, split="train")
    elif config.eval_split_ratio > 0:
        split = train.train_test_split(test_size=config.eval_split_ratio, seed=42)
        train, eval = split["train"], split["test"]
    else:
        eval = None
    
    return train, eval

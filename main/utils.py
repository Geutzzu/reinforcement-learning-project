from datasets import load_dataset


def load_datasets(config):
    train = load_dataset("parquet", data_files=config.train_dataset_path, split="train")
    
    # For SFT, we need 'completion' column. For GRPO, we need 'answer' for reward function.
    # Solution: Create 'completion' as a copy of 'answer' so both exist.
    if "answer" in train.column_names and "completion" not in train.column_names:
        train = train.add_column("completion", train["answer"])
    
    if config.eval_dataset_path:
        eval = load_dataset("parquet", data_files=config.eval_dataset_path, split="train")
        if "answer" in eval.column_names and "completion" not in eval.column_names:
            eval = eval.add_column("completion", eval["answer"])
    elif config.eval_split_ratio > 0:
        split = train.train_test_split(test_size=config.eval_split_ratio, seed=42)
        train, eval = split["train"], split["test"]
    else:
        eval = None
    
    return train, eval


import pandas as pd
import torch
import random
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import BartTokenizerFast, DataCollatorWithPadding
from transformers import BartModel, BartForConditionalGeneration, Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset, DatasetDict

# from transformers import AdamW
from transformers import get_scheduler

from tqdm.auto import tqdm

CHECKPOINT = 'facebook/bart-base'

def load_dataframe():
    data = pd.read_csv("../data/intermit/merged_dataset.tsv", sep='\t')
    data = data.iloc[:20000]

    return data

def prepare_dataset_dict(dataframe, tokenizer):
    
    X_tokenized = tokenizer(list(dataframe["toxic"].values), padding=True, truncation=True, return_tensors="pt")
    y_tokenized = tokenizer(list(dataframe["neutral1"].values), padding=True, truncation=True, return_tensors="pt")

    dataset = Dataset.from_dict({"input_ids": X_tokenized['input_ids'], 
                                 "attention_mask": X_tokenized['attention_mask'], 
                                 "label_ids": y_tokenized['input_ids']})
    
    # Split your dataset into training, validation, and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data = Dataset.from_dict(train_data)
    train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_data = Dataset.from_dict(train_data)
    test_data = Dataset.from_dict(test_data)
    validation_data = Dataset.from_dict(validation_data)

    datadict = DatasetDict({"train":train_data, "test":test_data, "validation":validation_data})
    datadict.save_to_disk(dataset_dict_path="../data/intermit/dataset-dict")

    return datadict

def main():
    dataframe = load_dataframe()

    tokenizer = BartTokenizerFast.from_pretrained(CHECKPOINT)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset_dict = prepare_dataset_dict(dataframe, tokenizer)

    train_dataloader = DataLoader(
        dataset_dict["train"], shuffle=True, batch_size=16, collate_fn=data_collator
    )

    model = BartForConditionalGeneration.from_pretrained(CHECKPOINT)
    optimizer = Adam(model.parameters(), lr=3e-5)
    
    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

if __name__ == "__main__":
    main()
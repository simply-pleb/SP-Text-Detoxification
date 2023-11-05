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
from datasets import load_from_disk

# from transformers import AdamW
from transformers import get_scheduler

from tqdm.auto import tqdm

import argparse

CHECKPOINT = 'facebook/bart-base'

def save_inputs_for_metrics():
    pass

def save_predicts_for_metrics():
    pass

def load_test_dataloader(data_collator):
    dataset = load_from_disk('data/intermit/dataset-dict')
    test_dataloader = DataLoader(
        dataset["test"], shuffle=True, batch_size=16, collate_fn=data_collator
    )
    return test_dataloader

def load_model():
    model = BartForConditionalGeneration.from_pretrained(CHECKPOINT)
    model.load_state_dict(torch.load("models/bart-paradetox"))
    return model

def main():
    parser = argparse.ArgumentParser(description='Predict a model using a sequence.')
    parser.add_argument('--pred', type=str, default=None,
                        help='The sequence to predict.')
    
    tokenizer = BartTokenizerFast.from_pretrained(CHECKPOINT)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = load_model()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    args = parser.parse_args()
    sequence_to_predict = args.pred
    if sequence_to_predict:
        tokenized = tokenizer(sequence_to_predict, padding=True, truncation=True, return_tensors="pt")
        # tokenized = tokenizer(sequence_to_predict, padding=True, truncation=True, return_tensors="pt")
        train_data = Dataset.from_dict(tokenized)
        dataloader = DataLoader(
            train_data, shuffle=False, batch_size=1, collate_fn=data_collator
        )
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(**batch)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("input:", sequence_to_predict)
            print("output:", text)
        return

    test_dataloader = load_test_dataloader(data_collator)    

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(**batch)

        inputs = batch['input_ids']
        with open('data/intermit/test-input-prediction/input.txt', 'a') as file:
            for item in inputs:
                text = tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                file.write(text + '\n')
        with open('data/intermit/test-input-prediction/output.txt', 'a') as file:
            for item in outputs:
                text = tokenizer.decode(item, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                file.write(text + '\n')



if __name__ == "__main__":
    main()
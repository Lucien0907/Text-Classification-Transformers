import os
import sys
import json
import logging
import traceback

from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import date, datetime
from huggingface_hub import login
import torch
import datasets
import evaluate
import pandas as pd
import numpy as np

# from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

logging.getLogger().setLevel(logging.INFO)

def main(token_hf, dataset_name, model_id, tokenize_args, train_args, lora_config=None):
    
    logging.info(torch.cuda.is_available())
    torch.cuda.empty_cache() 

    # log into hugginface via acces token
    login(token=token_hf, add_to_git_credential=True)

    # Load local dataset 
    raw_dataset = datasets.load_from_disk(f"./data/{dataset_name}").rename_column("label", "labels")
    # raw_dataset_train = datasets.load_dataset("csv", data_files=f"./data/{dataset_name}/{dataset_name}_train.csv")
    # raw_dataset_test = datasets.load_dataset("csv", data_files=f"./data/{dataset_name}/{dataset_name}_test.csv")
    # raw_dataset_train = raw_dataset_train.rename_column("condition", "labels").rename_column("review", "text")
    # raw_dataset_test = raw_dataset_test.rename_column("condition", "labels").rename_column("review", "text")
    print(f"Train dataset size: {len(raw_dataset['train'])}")
    print(f"Test dataset size: {len(raw_dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define tokenize function
    def tokenize(batch):
        return tokenizer(batch['text'], **tokenize_args)
    
    tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])
    # tokenized_dataset_train = raw_dataset_train.map(tokenize, batched=True, remove_columns=["text", "drugName"])
    # tokenized_dataset_test = raw_dataset_test.map(tokenize, batched=True, remove_columns=["text", "drugName"])

    # Prepare model labels
    labels = tokenized_dataset["train"].features["labels"].names
    print(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=len(labels), 
        label2id=label2id, 
        id2label=id2label
    )

    # apply lora adaptor if specified
    # if lora_config is not None:
    #     lora_config = LoraConfig(**lora_config)
    #     model = get_peft_model(model, lora_config)

    # set up training arguments
    timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    training_args = TrainingArguments(
        output_dir=f"results/{model_id}/{timestamp}/checkpoints/",
        logging_dir=f"results/{model_id}/{timestamp}/logs/",
        push_to_hub= False,
        **train_args
        )
    
    # Metric helper method
    metric = evaluate.load(train_args["metric_for_best_model"])
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    # Create trainer instance
    if model_id == "bert-base-uncased":
        params_add = {} 
    elif model_id == "distilbert/distilbert-base-uncased":
        params_add = {
            "tokenizer":tokenizer,
            "data_collator":data_collator,
        }
    print(params_add)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        **params_add
    )

    # Fine-tune the model
    trainer.train()

if __name__ == "__main__":
    try:
        # set up arguments
        parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-cfg", "--config", type=str, required=False, help="json file for training configuration")
        args = parser.parse_args()

        #load configuration file
        with open(args.config, "r") as f:
            cfg = json.load(f)

        for x in cfg:
            print(x)

        # run main function
        main(**cfg)
            
    except Exception as e:
        logging.error(traceback.format_exc())
        raise
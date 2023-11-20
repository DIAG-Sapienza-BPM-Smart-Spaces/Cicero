#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Credits -> Original script: https://github.com/project-baize/baize-chatbot/blob/main/finetune.py
import os
import argparse
import sys
import torch
import random
import json
import transformers

from datasets import load_dataset
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM


from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
)

from utils import *


parser = argparse.ArgumentParser()

parser.add_argument(
    "--exp_name", help="Experiment name", type=str, required=True
)

parser.add_argument(
    "--model_size",
    help="LLama model size, available options: 7b, 13b, 70b",
    type=str,
    default="7b",
)
args = parser.parse_args()

size = args.model_size

assert size in [
    "7b",
    "13b",
    "70b",
], f"Model size must be one of [7b, 13b, 70b]. You passed {size}"


# Parameters
MICRO_BATCH_SIZE = 16
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
LEARNING_RATE = 3e-4  # Karpathy constant
CUTOFF_LEN = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
]

OUTPUT_DIR = "checkpoints_new/{}_{}".format(size, args.exp_name)


# Load data
doc_path = ""

print('Selected train set', doc_path)
# Load the data from the CSV file into two separate dataframes, one for training and one for testing
train, test = load_into_dataframes(doc_path)
train, test =train[:2], test[:2]
# Convert the dataframes into HF datasets, which can be used to train transformers
data = dataframe_2_datasets(train, test)

# Load Model
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = False # world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

model = AutoModelForCausalLM.from_pretrained("GroNLP/gpt2-medium-italian-embeddings", torch_dtype=torch.float16)
total_params, params = 0, 0

tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-medium-italian-embeddings")

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    #target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)



config.save_pretrained(OUTPUT_DIR)

model = get_peft_model(model, config)
tokenizer.pad_token_id = 0

# Set the padding token to the end of sentence token
tokenizer.pad_token = tokenizer.eos_token


# Define a function that tokenizes a text input and returns a dictionary containing the tokenized input
# max_lenght can be changed according to preferences
def tok_func(x):
    return tokenizer(x["text"], max_length= 128,  truncation=True, padding="max_length") #TODO max_length=100

for n, p in model.model.named_parameters():
    if any([x in n for x in ["lora"]]):
        total_params += p.numel()
    params += p.numel()

print(
    "Total number of parameters: {}M, rate: {}%".format(
        total_params // 1000 / 1000, round(total_params / params * 100, 2)
    )
)

model.print_trainable_parameters()

device = "cuda"

def generate_text(model, tokenizer, prompt, length=50, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=length,max_length=length)
    generated_text = tokenizer.batch_decode(gen_tokens)
    return generated_text


# Data Preprocess
def generate_prompt(data_point):
    return data_point["input"]


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)


if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(tok_func, batched=True) #, remove_columns=['__index_level_0__', "text"])
    val_data = None

# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps= 8, #GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=100,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

model.save_pretrained(OUTPUT_DIR)


texts = ["My name is","yesterday i went to the park", "Ciao come stai?"
         "Con ricorso del", "All'udienza del"]
for text in texts:

    generated_text = generate_text(model, tokenizer, text, length= 50, do_sample=True)
    print(generated_text)
   






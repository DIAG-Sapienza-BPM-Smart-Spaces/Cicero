# ===========================================
# ||                                       ||
# ||Section 1: Importing modules           ||
# ||                                       ||
# ===========================================

import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import transformers
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from utils import *

# ===========================================
# ||                                       ||
# ||Section 2: checking gpu, choosing      ||
# ||             device, and model         ||
# ||                                       ||
# ===========================================

# Check GPU availability and get the device
print(check_gpu_availability())
device = getting_device()

# Workin_models return a list of models that work with this script and colab non-premium, pick one
model_nm = working_models()[0]

print('Selected model: ', model_nm)
# ===========================================
# ||                                       ||
# ||Section 3: Importing doc into dataset  ||
# ||                                       ||
# ===========================================

# Set the path to the CSV file
doc_path = "/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Merging/GLSA_merge_2_400.csv"

# Load the data from the CSV file into two separate dataframes, one for training and one for testing
train, test = load_into_dataframes(doc_path)
print_gpu_utilization()
print(len(train['text'].iloc[0])) # TODO elimina
# Convert the dataframes into HF datasets, which can be used to train transformers
ds = dataframe_2_datasets(train, test)

# ===========================================
# ||                                       ||
# ||Section 4: tokenization and collider   ||
# ||                                       ||
# ===========================================

# Load the tokenizer for the specified pre-trained language model
tokz = AutoTokenizer.from_pretrained(model_nm)

# Set the padding token to the end of sentence token
tokz.pad_token = tokz.eos_token

# Define a function that tokenizes a text input and returns a dictionary containing the tokenized input
# max_lenght can be changed according to preferences
def tok_func(x):
    return tokz(x["text"], truncation=True, max_length=10, padding="max_length") #TODO max_length=100

# Apply the tokenization function to the "train" and "test" datasets
# removing the original "text" column and adding the tokenized column
ds["train"] = ds["train"].map(tok_func, batched=True, remove_columns=['__index_level_0__', "text"])
ds["test"] = ds["test"].map(tok_func, batched=True, remove_columns=['__index_level_0__', "text"])

print((ds["train"][:10]))
# Define a data collator for the transformer that uses the same tokenizer as above and sets masked language modeling to false
data_collator = DataCollatorForLanguageModeling(tokenizer=tokz, mlm=False)
# ===========================================
# ||                                       ||
# ||Section 5: building the model          ||
# ||                                       ||
# ===========================================

# should be empty before the training
print_gpu_utilization()

# Load the pre-trained language model for causal language modeling with caching disabled
# and move it to the specified device (e.g. GPU)
model = AutoModelForCausalLM.from_pretrained(model_nm, use_cache=False).to(device)


# if still empty something went wrong
print_gpu_utilization()

# Set the hyperparameters for training the language model
training_args = TrainingArguments(
    model_nm, # The name of the pre-trained model to use
    evaluation_strategy="epoch", # The frequency at which to evaluate the model during training (in this case, at the end of each epoch)
    learning_rate=2e-5, # The learning rate to use for the optimizer during training
    weight_decay=0.01, # The weight decay to use for the optimizer during training
    num_train_epochs=15, # The number of epochs to train the model for
    per_device_train_batch_size=1, # The batch size to use for training on each device (in this case, 2)
    per_device_eval_batch_size=1, # The batch size to use for evaluation on each device (in this case, 2)
    gradient_accumulation_steps=1, # The number of gradient accumulation steps to use to offset the small batch size due to memory constraints
    gradient_checkpointing=True, # Whether to use gradient checkpointing to reduce memory usage during training
    fp16=True # Whether to use mixed precision training to speed up training and reduce memory usage
)

# Create a Trainer object that will be used to train the language model
trainer = Trainer(
    model=model, # The pre-trained language model to train
    args=training_args, # The hyperparameters for training the model
    train_dataset=ds["train"], # The dataset to use for training the model
    eval_dataset=ds["test"], # The dataset to use for evaluating the model during training
    data_collator=data_collator # The data collator to use for collating input data during training
)


# ===========================================
# ||                                       ||
# ||Section 5: training and saving         ||
# ||                                       ||
# ===========================================

# train the model
trainer.train()

# Set the output directory
output_dir = '/home/gueststudente/Giustizia/Processing/output_dir'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model and tokenizer to the output directory
trainer.save_model(output_dir)
tokz.save_pretrained(output_dir)
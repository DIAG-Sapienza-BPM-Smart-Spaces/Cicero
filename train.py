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
from optimum.bettertransformer import BetterTransformer
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from utils import *

from fairscale.nn import Pipe
from fairscale.nn.pipe.balance import balance_by_time
from transformers import AutoModel

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# ===========================================
# ||                                       ||
# ||Section 2: checking gpu, choosing      ||
# ||             device, and model         ||
# ||                                       ||
# ===========================================

# Check GPU availability and get the device
#print(check_gpu_availability())
device = getting_device()

# Workin_models return a list of models that work with this script and colab non-premium, pick one
model_nm = working_models()[3]

print('Selected model: ', model_nm)
# ===========================================
# ||                                       ||
# ||Section 3: Importing doc into dataset  ||
# ||                                       ||
# ===========================================

# Set the path to the CSV file

doc_path = "merge_22_04_3.csv"
print('Selected train set', doc_path)
# Load the data from the CSV file into two separate dataframes, one for training and one for testing
train, test = load_into_dataframes(doc_path)
#print_gpu_utilization()
#print(len(train['text'].iloc[0])) # TODO elimina
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


#print(list(ds.keys()))
# GETTING MAX LENGTH
max_length = get_max_lenghts(get_list_of_lengths(ds["train"]["text"], tokz))

# Define a function that tokenizes a text input and returns a dictionary containing the tokenized input
# max_lenght can be changed according to preferences
def tok_func(x):
    return tokz(x["text"], truncation=True, max_length=max_length, padding="max_length") #TODO max_length=100

# Apply the tokenization function to the "train" and "test" datasets
# removing the original "text" column and adding the tokenized column
ds["train"] = ds["train"].map(tok_func, batched=True, remove_columns=['__index_level_0__', "text"])
ds["test"] = ds["test"].map(tok_func, batched=True, remove_columns=['__index_level_0__', "text"])

#print((ds["train"][:10]))
# Define a data collator for the transformer that uses the same tokenizer as above and sets masked language modeling to false
data_collator = DataCollatorForLanguageModeling(tokenizer=tokz, mlm=False)
# ===========================================
# ||                                       ||
# ||Section 5: building the model          ||
# ||                                       ||
# ===========================================

# should be empty before the training
#print_gpu_utilization()

# Load the pre-trained language model for causal language modeling with caching disabled
# and move it to the specified device (e.g. GPU)
model = AutoModelForCausalLM.from_pretrained(model_nm)

#sample = torch.FloatTensor(ds['train'][0]['input_ids'])
#partitions = torch.cuda.device_count()
#print(partitions)
#balance = balance_by_time(partitions, model.parameters(), sample)
#print(balance)
#model = Pipe(model, balance)
#model_hf = AutoModelForCausalLM.from_pretrained("gpt2",device_map="auto")
#model = BetterTransformer.transform(model_hf, keep_original_model=True)
#model = BetterTransformer.transform(model, keep_original_model=False)
#model = torch.nn.DataParallel(model)
#model = model.to(device)


# if still empty something went wrong
#print_gpu_utilization()

# Set the hyperparameters for training the language model
training_args = TrainingArguments(
    model_nm, # The name of the pre-trained model to use
    evaluation_strategy="epoch", # The frequency at which to evaluate the model during training (in this case, at the end of each epoch)
    learning_rate=2e-5, # The learning rate to use for the optimizer during training
    weight_decay=0.01, # The weight decay to use for the optimizer during training
    num_train_epochs=4, # The number of epochs to train the model for
    per_device_train_batch_size=1, # The batch size to use for training on each device (in this case, 2)
    per_device_eval_batch_size=1, # The batch size to use for evaluation on each device (in this case, 2)
    remove_unused_columns=False,
    save_strategy="epoch",
    save_total_limit=10,
    sharded_ddp='zero_dp_2', # Full GPU parallelism
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
# ||Section 6: Perplexity                  ||
# ||                                       ||
# ===========================================

# reference => https://huggingface.co/docs/transformers/perplexity

# Tokenize the test data and convert it to PyTorch tensors
text = "\n\n".join(test["text"]) # concatenate all the text in the test set
encodings = tokz(text, return_tensors="pt") # get its integer id 

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over input tokens.
        # Multiply it with trg_len to get the summation instead of average.
        # We will take average over all the tokens to get the true average
        # in the last step of this example.
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print("evaluation: ",ppl)


# ===========================================
# ||                                       ||
# ||Section 7: training and saving         ||
# ||                                       ||
# ===========================================

# train the model
trainer.train()

# Set the output directory
output_dir = 'output2'

# Create the output directory if it doesn't exist
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

# Save the model and tokenizer to the output directory
trainer.save_model(output_dir)
tokz.save_pretrained(output_dir)


# ===========================================
# ||                                       ||
# ||Section 7: Perplexity                  ||
# ||                                       ||
# ===========================================

# reference => https://huggingface.co/docs/transformers/perplexity

# Tokenize the test data and convert it to PyTorch tensors
text = "\n\n".join(test["text"]) # concatenate all the text in the test set
encodings = tokz(text, return_tensors="pt") # get its integer id 

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over input tokens.
        # Multiply it with trg_len to get the summation instead of average.
        # We will take average over all the tokens to get the true average
        # in the last step of this example.
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print("evaluation: ",ppl)




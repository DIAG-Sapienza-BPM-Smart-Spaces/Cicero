from train import do_train_and_save_model, compute_perplexity

from utils import load_and_prepare_test

import torch 
from transformers import  AutoTokenizer, AutoModelForCausalLM 


# MODEL_LIST =
"""
LorenzoDeMattei/GePpeTto
GroNLP/gpt2-small-italian
GroNLP/gpt2-medium-italian-embeddings
microsoft/phi-1_5
meta-llama/Llama-2-7b
"""

# Choose the models to train
MODEL_LIST = []

# Choose the csv files to use for training
CSV_LIST = [ ]

# Insert your token here
TOKEN_HF = ""

# load the test set for the perplexity
test_dataset = load_and_prepare_test("Perplexity.csv")

# keep track of the perplexity per model id
perplexity_dict = {model_id:100_000 for model_id in MODEL_LIST} #latest perplexity
full_perplexity_dict = {model_id:[] for model_id in MODEL_LIST} #full perplexity history

for model_id in MODEL_LIST:
    # init variables for training resume for every new model
    continue_training = False
    previous_path = ""
    # initial perplexity -- baseline
    initial_perplexity = 9999
    try:
        initial_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,  device_map="auto", token=TOKEN_HF)
        initial_tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN_HF)
        initial_perplexity = compute_perplexity(model=initial_model, tokenizer=initial_tokenizer, test=test_dataset)
        perplexity_dict[model_id] = initial_perplexity
        full_perplexity_dict[model_id].append(initial_perplexity)

    except Exception as e:
        print(e)
        print("Something went wrong with initial perplexity")

    print(f"Initial perplexity for {model_id} is {initial_perplexity}")

    for path in CSV_LIST:
        print(f"Training {model_id} with file {path.split('/')[-1].split('.')[0]} MB")

        if continue_training:
            # the perplexity is decreasing, continue with new chunk of data
            trained_model, tokenizer = do_train_and_save_model(model_id=model_id, dataset_path=path, continue_training=True, model_path=previous_path)
        else:
            # first tarin run per model
            trained_model, tokenizer = do_train_and_save_model(model_id=model_id, dataset_path=path)
        
        perplexity = compute_perplexity(model=trained_model, tokenizer=tokenizer, test=test_dataset)

        # check whether the perplexity is decreasing
        if perplexity >= perplexity_dict[model_id]:
            print('perplexity >= perplexity_dict')
            #break
        else:
            # update the trainig variables and the perplexity dict
            perplexity_dict[model_id] = perplexity
            continue_training = True
            previous_path = f"save_{model_id.split('/')[1]}_{path.split('/')[-1].split('.')[0]}MB"
        
        # update anyways full perplexity
        full_perplexity_dict[model_id].append(perplexity)

# display results
print()
print("="*30)
print("BEST PERPLEXITY RESULTS")
print(perplexity_dict)        
print("FULL PERPLEXITY RESULTS")
print(full_perplexity_dict)  
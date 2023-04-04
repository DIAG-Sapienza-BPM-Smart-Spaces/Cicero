# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||    
# ||                                       ||
# ===========================================
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# ===========================================
# ||                                       ||
# ||  Section 2: utils functions for gpu   ||
# ||             and device                ||    
# ||                                       ||
# ===========================================

def check_gpu_availability():
    # Check if CUDA is available
    print(f"Cuda is available: {torch.cuda.is_available()}")

def getting_device(gpu_prefence=True) -> torch.device:
    """
    This function gets the torch device to be used for computations, 
    based on the GPU preference specified by the user.
    """
    
    # If GPU is preferred and available, set device to CUDA
    if gpu_prefence and torch.cuda.is_available():
        device = torch.device('cuda')
    # If GPU is not preferred or not available, set device to CPU
    else: 
        device = torch.device("cpu")
    
    # Print the selected device
    print(f"Selected device: {device}")
    
    # Return the device
    return device

# Define a function to print GPU memory utilization
def print_gpu_utilization():
    # Initialize the PyNVML library
    nvmlInit()
    # Get a handle to the first GPU in the system
    handle = nvmlDeviceGetHandleByIndex(0)
    # Get information about the memory usage on the GPU
    info = nvmlDeviceGetMemoryInfo(handle)
    # Print the GPU memory usage in MB
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# Define a function to print training summary information
def print_summary(result):
    # Print the total training time in seconds
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # Print the number of training samples processed per second
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # Print the GPU memory utilization
    print_gpu_utilization()

# ===========================================
# ||                                       ||
# ||  Section 3: utils function for listing||
# ||             working model on colab    ||
# ||             (not premium version)     ||    
# ||                                       ||
# ===========================================

def working_models() -> list:
    # Define a list of working models
    models = ["GroNLP/gpt2-medium-italian-embeddings","sberbank-ai/mGPT","GroNLP/gpt2-small-italian", "GroNLP/gpt2-medium-italian-embeddings", "facebook/xglm-1.7B",
              "facebook/xglm-7.5B",  "facebook/xglm-564M" ]
    
    # Return the list of models
    return models

# ===========================================
# ||                                       ||
# ||  Section 4: utils function to         ||  
# ||             load datasets into        ||  
# ||             inputs for the tokenizer  ||    
# ||                                       ||
# ===========================================

def load_into_dataframes(path_2_corpus: str, reduce_size = 0, split_size = 0.2, random_state_chosen =42 ) -> pd.core.frame.DataFrame:
    '''
    INPUT => The corpus should be a csv with only one column containing the documents (civil cases)
    OUTPUT => two pandas train and test dataframes with corpus under column TEXT

    '''
    # Read the csv file into a dataframe
    df = pd.read_csv(path_2_corpus,  engine='python', index_col=False)

    # Rename the column to "text"
    df = df.rename(columns={df.columns[0]: "text"})

    # If specified, reduce the size of the dataframe
    if reduce_size:
        df = df.head(reduce_size)

    # Split the dataframe into train and test sets
    return  train_test_split(df, test_size=split_size, random_state= random_state_chosen)

def dataframe_2_datasets(train_df, test_df):
    ''' 
    INPUT => the two pandas dataframe
    OUTOUT => a dataset with two dataframe, each of them must be tokenized => once tokenized can be used as input for the model
    '''
    # Convert the training dataframe to a Hugging Face Dataset
    ds_train = Dataset.from_pandas(train_df)

    # Convert the test dataframe to a Hugging Face Dataset
    ds_test = Dataset.from_pandas(test_df)

    # Return a DatasetDict containing the training and test datasets
    return DatasetDict({"train": ds_train, "test": ds_test})




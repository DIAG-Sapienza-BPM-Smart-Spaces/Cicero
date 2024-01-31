import os
import sys
import torch
import transformers

from tqdm import tqdm

from transformers import  AutoTokenizer, AutoModelForCausalLM 


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from utils import load_into_dataframes, dataframe_2_datasets



def do_train_and_save_model(model_id, dataset_path, continue_training=False, model_path=""):
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

    OUTPUT_DIR =  f"save_{model_id.split('/')[1]}_{dataset_path.split('/')[-1].split('.')[0]}MB"


    # Load data
    DOC_PATH = dataset_path

    print('Selected train set', DOC_PATH)
    # Load the data from the CSV file into two separate dataframes, one for training and one for testing
    train, test = load_into_dataframes(DOC_PATH)

    data = dataframe_2_datasets(train, test)

    # Load Model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = False # world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


    # come reference noi utilizziamo per questo file di train LORA (Low-Rank Adaptation of LLM)
    # che ottimizza l'utilizzo della memoria e dei parametri trainati senza compromettere le performance

    # se dà OOM, prova con load_in_8bit=True o load_in_4bit=True (Quantizzazione in 8 o 4 bit)
    # così riduci il peso in memoria a discapito della precisione
    # in caso probailmente sarà necessario togliere il commento a prepare_model_for_kbit_training
    if continue_training: 
        print("resume training from", model_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        except Exception as e:
            print(e)
            print("Something went wrong, maybe safetensors?")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    total_params, params = 0, 0

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Se non usiamo la quantizzazione lasciamo perdere
    #model = prepare_model_for_kbit_training(model)


    # Config di lora
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        #target_modules=TARGET_MODULES, # Molti modelli prendono questo in automatico 
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0

    # Set the padding token to the end of sentence token
    tokenizer.pad_token = tokenizer.eos_token


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

    # questa dà problemi: model.save_pretrained(OUTPUT_DIR)
    # per salvare usiamo questa
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)

    def generate_text(model, tokenizer, prompt, length=50, do_sample=True):
        device="cuda"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=length,max_length=length)
        generated_text = tokenizer.batch_decode(gen_tokens)
        return generated_text


    texts = [
            "Con ricorso del", "All'udienza del", "In data", "I coniugi comparivano", "La parte, in presenza del giudice"
            ]
    try:
        for text in texts:
            generated_text = generate_text(merged_model, tokenizer, text, length= 50, do_sample=True)
            print(generated_text, "!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    except Exception as e: 
        print("qualcosa non ha funzionato nella generazione del testo")
        print(e)
    return model, tokenizer

def compute_perplexity(model, tokenizer, test):
    device = "cuda"
    text = "\n\n".join(test["text"]) # concatenate all the text in the test set
    encodings = tokenizer(text, return_tensors="pt") # get its integer id 
    max_length = 512 ####TOGLIERE, FUNZIONA SOLO CON LARGE-BART
    #max_length = model.config.n_positions
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
    return ppl


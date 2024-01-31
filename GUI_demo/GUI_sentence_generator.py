import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the device to use
device = 'mps' #'cuda' if torch.cuda.is_available() else 'cpu'

# Load the saved model and tokenizer
model_path = 'DIAG-PSSeng/cicero_v2-phi1.5'
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_text(prompt):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text using the model
    output = model.generate(input_ids=input_ids, max_length=500, do_sample=True)

    # Decode the output to get the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text
    return generated_text

# Create the Gradio interface
iface = gr.Interface(
    generate_text,
    inputs=gr.inputs.Textbox(label="Prompt"),
    outputs=gr.outputs.Textbox(label="Generated Text")
)

# Launch the interface
iface.launch()

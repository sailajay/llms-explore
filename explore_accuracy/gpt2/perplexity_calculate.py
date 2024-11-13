import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch
import math

# Initialize the model and tokenizer for PPL evaluation
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load WikiText-2 test dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10]") 

# Function to send prompt to your inference engine
def get_generated_text(prompt, api_url):
    response = requests.post(api_url, json={"input_text": prompt})
    if response.status_code == 200:
        return response.json().get("generated_text")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to calculate perplexity given the generated text
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    # Compute perplexity
    perplexity = math.exp(loss)
    return perplexity

# API endpoint for your inference engine
inference_url = "http://localhost:8000/generate"

# Measure perplexity for generated texts
total_perplexity = 0
count = 0
print(f"length of dataaset: {len(dataset)}  ")
for i, sample in enumerate(dataset['text']):
    if sample.strip():  # Skip empty lines
        generated_text = get_generated_text(sample, inference_url)
        print("--------------------------------")
        print(f"sample-{i+1}:  {sample[100:]}")
        print(f"generated text: {generated_text[100:]}")
        print("--------------------------------")
        if generated_text:
            ppl = calculate_perplexity(model, tokenizer, generated_text)
            total_perplexity += ppl
            count += 1
            print(f"Sample {i + 1}: Perplexity = {ppl}")

# Calculate average perplexity over the evaluated samples
average_perplexity = total_perplexity / count if count > 0 else float('inf')
print(f"Average Perplexity on {count} samples of WikiText-2: {average_perplexity}")

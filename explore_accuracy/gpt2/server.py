from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "GPT-2 Inference API"}

# Load the model and tokenizer
model_name = "gpt2"  # You can change this to "gpt2-medium" or other variants
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).eval()

# Move the model to GPU if available
device = torch.device("cpu")
model.to(device)
model.config.pad_token_id = model.config.eos_token_id

# Define a request body model
class GenerateRequest(BaseModel):
    input_text: str
    max_length: int = 200  # Default max length for generation

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    input_text = request.input_text
    max_length = request.max_length

    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the input text correctly
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Extract input_ids and attention_mask
    input_ids = inputs['input_ids']         # Get the input_ids from the tokenizer output
    attention_mask = inputs['attention_mask']  # Get the attention_mask from the tokenizer output

    # Generate text using the model
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=1.0,  # Adjust for creativity
            top_p=0.95,       # Nucleus sampling
            top_k=50          # Beam search size
        )

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"generated_text": generated_text}

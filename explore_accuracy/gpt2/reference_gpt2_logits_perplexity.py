from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=None)
ref_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=None)

acts = {}

# register hooks (to be called during forward())
def get_act(name):
    def hook(model, input, output):
        acts[f"{name}_in"] = input
        acts[f"{name}_out"] = output
    return hook
for k, v in dict(ref_model.named_modules()).items():
    v.register_forward_hook(get_act(k))

#Read prompts from file
prompt_file_path = 'prompts.txt'
with open(prompt_file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
prompts = lines[:1000]

# Initialize lists to store perplexities and log probabilities
all_perplexities = []
all_log_probs = []

for prompt_idx, prompt in enumerate(prompts):
    inputs = tokenizer(prompt)
    output_ids = torch.Tensor(inputs.input_ids).unsqueeze(0).to(torch.int32)

    end_generated_tokens=[]
    token_log_probs_for_prompt=[]

    # generate 20 tokens in total, 1 at a time
    for i in range(20):
        # generate 1 token
        output_ids = ref_model.generate(output_ids, max_new_tokens=1, attention_mask=torch.ones(output_ids.shape, device=output_ids.device), pad_token_id=tokenizer.eos_token_id)
        #output_ids = ref_model.generate(output_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        
        # this is the logits of the most recent iteration
        # Capture the logits after the generation step (from the hook)
        logits = acts.get("lm_head_out", None)
        
        # Save logits to file after each iteration
        # if logits is not None:
        #     logits_filename = f"logits_iteration_{i + 1}.pt"
        #     torch.save(logits, logits_filename)
        #     print(f"Saved logits for iteration {i + 1} to {logits_filename}")
        
        # Decode the token ID to a human-readable string
        last_token_id = output_ids[0, -1].item()
        generated_token = tokenizer.decode(last_token_id)
        end_generated_tokens.append(generated_token)
        
        #Get log prob of the generated token with log_softmax
        # Get logits for the last token generated (batch_size=1)
        #last_token_logits = logits[0, -1, :]  
        last_token_logits = acts["lm_head_out"][0, -1, :]
    
        # Apply log_softmax to get log probabilities for each token in the vocabulary
        log_probs = F.log_softmax(last_token_logits, dim=-1)
        # Get the selected token's ID (this is the token with the highest log probability)
        selected_token_id = torch.argmax(log_probs).item()

        #token_ids = torch.argmax(log_probs, dim=-1)
        #logprob = log_probs[token_ids].item()


        # Get the log probability of the selected token
        selected_token_log_prob = log_probs[selected_token_id].item()
        token_log_probs_for_prompt.append(selected_token_log_prob)
        #token_log_probs_for_prompt.append(logprob)
        # replace generated output token with our own (in this case we use our own softmax)
        output_ids[0, -1] = torch.argmax(acts["lm_head_out"][0, -1, :]).item()

    #calculate perplexity for the prompt 
    token_log_probs_for_prompt = np.array(token_log_probs_for_prompt)
    perplexity = np.exp(-np.mean(token_log_probs_for_prompt))
    #all_perplexities.append(perplexity)
    # Add log probabilities to overall list for final perplexity computation
    all_log_probs.extend(token_log_probs_for_prompt)
    print(f"Generated tokens for prompt {prompt_idx +1}: {end_generated_tokens}") 
    print(f"Perplexity for prompt {prompt_idx + 1}: {perplexity}")
    print("_________________________________________________________________")

# Calculate overall perplexity for the entire dataset
overall_perplexity = np.exp(-np.mean(all_log_probs))  

print(f"Overall Perplexity for the dataset: {overall_perplexity}")


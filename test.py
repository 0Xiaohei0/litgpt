import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

state_dict = torch.load('out/lora_merged/Huggingface/Rana.pth')
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('/workspace/litgpt/checkpoints/stabilityai/stablelm-zephyr-3b')
model = AutoModelForCausalLM.from_pretrained('/workspace/litgpt/checkpoints/stabilityai/stablelm-zephyr-3b', state_dict=state_dict)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the selected device
model.to(device)

# Set the model to evaluation mode
model.eval()

# Prepare an input prompt
input_prompt = "<|user|>Introduce yourself.<|assistant|>"

# Encode the input prompt
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

# Move the encoded input to the same device as the model
input_ids = input_ids.to(device)

# Generate a response, adjusting parameters as needed
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated response
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)


model = model.half()

# Save your model
model.save_pretrained('./Rana', torch_dtype=torch.float16)

# Save the tokenizer
tokenizer.save_pretrained('./Rana')
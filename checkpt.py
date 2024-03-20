import torch

# Load the .pt file
file_path = 'data\\custom_data\\train.pt'
contents = torch.load(file_path)

# Ensure the loaded contents are indeed a list
if isinstance(contents, list):
    print(f"Loaded a list with {len(contents)} items.")
    
    # Iterate over each item in the list
    for idx, item in enumerate(contents):
        print(f"Item {idx}: Type {type(item)}")
        
        # If the item is a tensor, you might want to print its shape
        if torch.is_tensor(item):
            print(f"  Tensor shape: {item.shape}")
        
        # If the item is a model, you might print a summary or its architecture
        elif isinstance(item, torch.nn.Module):
            print(f"  Model architecture: {item}")
            
        # If the item is a dictionary, print its keys
        elif isinstance(item, dict):
            print(f"  Dictionary with keys: {list(item.keys())}")
            # Assuming 'item' is the dictionary from your message
            print(f"Instruction: {item['instruction']}")
            print(f"Input: {item['input']}")
            print(f"Output: {item['output']}")
            print(f"Input IDs: {item['input_ids'][:50]}")  # Print first 10 for brevity
            print(f"Labels: {item['labels'][:50]}")  # Print first 10 for brevity
            
        # Handle other types as needed
        else:
            print(f"  Content: {item}")

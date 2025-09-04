from safetensors import safe_open
import os

checkpoint_path = "/home/haoming/Bagel/results/checkpoints/0000160"
model_state_dict_path = os.path.join(checkpoint_path, "model.safetensors")

try:
    with safe_open(model_state_dict_path, framework="pt", device="cpu") as f:
        print("Available keys:", f.keys())
        print("Metadata:", f.metadata())
except Exception as e:
    print(f"Error reading file: {e}")
    
    # Check file size and basic info
    if os.path.exists(model_state_dict_path):
        file_size = os.path.getsize(model_state_dict_path)
        print(f"File size: {file_size} bytes")
        if file_size == 0:
            print("File is empty!")
        elif file_size < 1024:
            print("File is suspiciously small")
    else:
        print("File doesn't exist")
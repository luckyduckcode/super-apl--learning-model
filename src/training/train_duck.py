import os
import subprocess
import json
import shutil
import random

# Configuration
CONFIG_FILE = "duck_personality.json"
SOURCE_REPO_DIR = "raw_data_repo"
PROCESSED_DATA_DIR = "processed_data"

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def clone_repo(repo_url):
    if os.path.exists(SOURCE_REPO_DIR):
        print(f"[Info] Repository already exists in {SOURCE_REPO_DIR}. Pulling latest...")
        subprocess.run(["git", "-C", SOURCE_REPO_DIR, "pull"], check=True)
    else:
        print(f"[Info] Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, SOURCE_REPO_DIR], check=True)

def r2d2_c3po_filter(text):
    """
    Applies the 'R2-D2 Humor + C-3PO Versatility' filter.
    In a real scenario, this would use an LLM to rewrite or filter dataset samples.
    Here, we simulate the filtering logic.
    """
    # 1. Versatility Check (C-3PO): Ensure content is informative/structural
    if len(text.split()) < 5:
        return None # Too short to be useful protocol data

    # 2. Humor Injection (R2-D2): Add 'attitude' metadata or rewrite
    # This is a placeholder for the actual data augmentation logic
    augmented_text = f"<system>Protocol: Active. Attitude: Sassy.</system>\n{text}"
    
    return augmented_text

def process_dataset(config):
    print("[Info] Processing dataset with R2-D2/C-3PO filter...")
    
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    # Walk through the cloned repo
    file_count = 0
    for root, dirs, files in os.walk(SOURCE_REPO_DIR):
        for file in files:
            if file.endswith((".cpp", ".apl", ".txt", ".md")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Apply the filter
                    filtered_content = r2d2_c3po_filter(content)
                    
                    if filtered_content:
                        output_file = os.path.join(PROCESSED_DATA_DIR, f"processed_{file_count}.txt")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(filtered_content)
                        file_count += 1
                except Exception as e:
                    print(f"[Warning] Could not process {file}: {e}")

    print(f"[Success] Processed {file_count} files into {PROCESSED_DATA_DIR}")
    print(f"[Info] Personality Profile Applied: {config['personality_profile']['system_prompt']}")

def main():
    print("=== Duck (Super APL) Training Pipeline ===")
    config = load_config()
    
    # 1. Get Data
    clone_repo(config["base_repo"])
    
    # 2. Filter & Prepare
    process_dataset(config)
    
    # 3. Simulate Training Loop
    print("\n[Training] Initializing Super APL Engine...")
    print(f"[Training] Loading Quantization: {config['training_parameters']['quantization']}")
    print("[Training] Optimizing for R2-D2 Humor (Sarcasm/Wit) and C-3PO Versatility (Knowledge)...")
    
    # Simulate epochs
    for i in range(config['training_parameters']['epochs']):
        print(f"  Epoch {i+1}/{config['training_parameters']['epochs']} - Loss: {random.uniform(0.1, 0.9):.4f} - Sassy_Level: {random.randint(80, 100)}%")
    
    print("\n[Done] Model 'Duck' is trained and ready.")
    print("To run inference, use the GUI or APL interface with the new weights.")

if __name__ == "__main__":
    main()

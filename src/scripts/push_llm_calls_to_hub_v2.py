import os
import sqlite3
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import gc

load_dotenv()


def create_messages_column(row):
    """Transform a row into ChatML messages format"""
    # Handle potential None values
    system_prompt = row['system_prompt'] if row['system_prompt'] else ""
    human_prompt = row['human_prompt'] if row['human_prompt'] else ""
    assistant_response = row['assistant_response'] if row['assistant_response'] else ""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
        {"role": "assistant", "content": assistant_response}
    ]


def main():
    # Check for HF token - first try env var, then cached token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        # Try to read from cached token file
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                hf_token = f.read().strip()
        else:
            raise ValueError("HF_TOKEN environment variable not set and no cached token found")
    
    # Database path - updated for new codebase
    db_path = "src/data/distill_graph.db"
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    print(f"Connecting to database: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Get total count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM distill_llm_calls")
        total_records = cursor.fetchone()[0]
        print(f"Found {total_records} records")
        
        if total_records == 0:
            print("No records to upload")
            return
        
        # Read data with ORDER BY RANDOM() directly in SQL (more efficient)
        print("Reading and randomizing data...")
        dataframe = pd.read_sql_query(
            "SELECT * FROM distill_llm_calls ORDER BY RANDOM()", 
            conn
        )
        
        # Process messages more efficiently
        print("Creating messages column...")
        messages = []
        for idx in range(len(dataframe)):
            row = dataframe.iloc[idx]
            messages.append([
                {"role": "system", "content": row['system_prompt'] or ""},
                {"role": "user", "content": row['human_prompt'] or ""},
                {"role": "assistant", "content": row['assistant_response'] or ""}
            ])
        
        dataframe['messages'] = messages
        
        # Force garbage collection
        gc.collect()
        
        # Convert to HuggingFace Dataset with memory mapping
        print("Converting to HuggingFace Dataset...")
        dataset = Dataset.from_pandas(dataframe, preserve_index=False)
        
        # Upload to HuggingFace Hub
        repo_id = "UWV/wim_instruct_signaalberichten_to_jsonld_agent_steps"
        print(f"Uploading to HuggingFace Hub: {repo_id}")
        
        dataset.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
            split="train",
            private=False,
            max_shard_size="500MB"  # Split into smaller shards
        )
        
        print(f"✓ Successfully uploaded {len(dataframe)} records to {repo_id}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Parallel distillation pipeline for text-to-knowledge-graph with LLM logging.
Optimized for the UWV/wim-synthetic-data-rd dataset format.
"""

import json
import yaml
import time
import shutil
import sqlite3
import argparse
import socket
import os
import sys
from datetime import datetime, timedelta
import random
from pathlib import Path
import fcntl  # For file-based locking

from graph import TextToKGState, TextToKGPipeline
from graph.utils import azure_llm
from graph.prompts import (
    ENTITY_EXTRACTION_HUMAN_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT,
    RETRIEVE_SCHEMA_ORG_SYSTEM_PROMPT,
    TRANSFORM_TO_KG_HUMAN_PROMPT,
    TRANSFORM_TO_KG_SYSTEM_PROMPT,
    ADD_LABELS_HUMAN_PROMPT,
    ADD_LABELS_SYSTEM_PROMPT,
)

from openai import BadRequestError
from datasets import load_dataset
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()


def retry_on_locked(func):
    """Retry decorator for database operations that may encounter locks"""
    def wrapper(*args, **kwargs):
        attempt = 0
        base_delay = 0.1  # Start with 100ms
        max_delay = 5.0  # Cap at 5 seconds
        max_attempts = 30  # Stop after 30 attempts (roughly 2-3 minutes total)

        while attempt < max_attempts:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                if "locked" in error_msg or "busy" in error_msg:
                    # Exponential backoff with jitter, capped at max_delay
                    delay = min(
                        base_delay * (2**attempt), max_delay
                    ) + random.uniform(0, 0.1)
                    
                    if attempt % 5 == 0:  # Log every 5 attempts
                        print(
                            f"    ⚠ Database locked/busy, retrying in {delay:.3f}s "
                            f"(attempt {attempt + 1}/{max_attempts})"
                        )
                    
                    time.sleep(delay)
                    attempt += 1
                    
                    # If we're getting close to max attempts, try a WAL checkpoint
                    if attempt == max_attempts - 5:
                        try:
                            print("    ⚠ Attempting WAL checkpoint to clear locks...")
                            temp_conn = sqlite3.connect(args.db_path if 'args' in globals() else 'src/data/distill_graph.db')
                            temp_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                            temp_conn.close()
                        except:
                            pass  # Ignore checkpoint errors
                    
                    continue
                raise
        
        # If we've exhausted all attempts
        raise sqlite3.OperationalError(
            f"Database remained locked after {max_attempts} attempts. "
            "Consider reducing the number of concurrent workers."
        )
    return wrapper


def init_tables(conn, cursor, add_labels=False):
    """Initialize tables using existing connection"""
    cursor.execute("BEGIN IMMEDIATE")
    try:
        # Core distillation tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS n1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_id TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                input_text TEXT,
                output_text TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS n2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_id TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                input_text TEXT,
                output_text TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS n3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_id TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                input_text TEXT,
                output_text TEXT,
                err_msg_1 TEXT,
                err_msg_2 TEXT,
                err_msg_3 TEXT,
                err_msg_4 TEXT,
                err_msg_5 TEXT,
                err_out_1 TEXT,
                err_out_2 TEXT,
                err_out_3 TEXT,
                err_out_4 TEXT,
                err_out_5 TEXT,
                validation_failed BOOLEAN DEFAULT FALSE
            )
        """)

        if add_labels:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS n5 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_id TEXT,
                    system_prompt TEXT,
                    user_prompt TEXT,
                    input_text TEXT,
                    output_text TEXT,
                    labels TEXT
                )
            """)

        # Work coordination tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_queue (
                batch_id INTEGER PRIMARY KEY,
                start_idx INTEGER NOT NULL,
                end_idx INTEGER NOT NULL,
                worker_id TEXT,
                status TEXT DEFAULT 'pending',
                claimed_at TIMESTAMP,
                completed_at TIMESTAMP,
                UNIQUE(start_idx, end_idx)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_status (
                unique_id TEXT PRIMARY KEY,
                processing_complete BOOLEAN DEFAULT FALSE,
                labels_added BOOLEAN DEFAULT FALSE,
                processed_at TIMESTAMP,
                worker_id TEXT
            )
        """)

        # LLM calls logging table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS distill_llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                call_name TEXT,
                system_prompt TEXT,
                human_prompt TEXT,
                assistant_response TEXT,
                created_at TIMESTAMP,
                worker_id TEXT,
                model_name TEXT
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_n1_unique_id ON n1(unique_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_n2_unique_id ON n2(unique_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_n3_unique_id ON n3(unique_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_status_complete ON processing_status(processing_complete)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_source ON distill_llm_calls(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_name ON distill_llm_calls(call_name)")
        if add_labels:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_n5_unique_id ON n5(unique_id)")

        cursor.execute("COMMIT")
    except Exception:
        cursor.execute("ROLLBACK")
        raise


def claim_batch(cursor, conn, worker_id, batch_size, total_size, db_path):
    """Atomically claim a batch of work using file-based locking"""
    # Use a lock file to ensure only one worker can claim at a time
    lock_file_path = f"{db_path}.claim_lock"
    
    # Create lock file if it doesn't exist
    Path(lock_file_path).touch()
    
    with open(lock_file_path, 'w') as lock_file:
        # Acquire exclusive file lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Now we have exclusive access to claim batches
            return _claim_batch_with_lock(cursor, conn, worker_id, batch_size, total_size)
        finally:
            # Release the file lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _claim_batch_with_lock(cursor, conn, worker_id, batch_size, total_size):
    """Internal function to claim batch - must be called with file lock held"""
    # Try to claim an existing pending batch
    @retry_on_locked
    def try_claim_pending():
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # First, find a pending batch
            cursor.execute("""
                SELECT batch_id FROM work_queue 
                WHERE status = 'pending' 
                ORDER BY batch_id
                LIMIT 1
            """)
            pending = cursor.fetchone()
            
            if not pending:
                cursor.execute("ROLLBACK")
                return None
                
            batch_id = pending[0]
            
            # Now claim it with a second check to ensure it's still pending
            cursor.execute("""
                UPDATE work_queue 
                SET worker_id = ?, status = 'in_progress', claimed_at = ?
                WHERE batch_id = ? AND status = 'pending'
                RETURNING batch_id, start_idx, end_idx
            """, (worker_id, datetime.now().isoformat(), batch_id))

            result = cursor.fetchone()
            if result:
                cursor.execute("COMMIT")
                print(f"    ✓ Worker {worker_id} claimed batch {result[0]} (rows {result[1]}-{result[2]})")
                return result
            else:
                # Someone else got it first
                cursor.execute("ROLLBACK")
                return None
        except Exception as e:
            cursor.execute("ROLLBACK")
            print(f"    ❌ Error claiming batch: {e}")
            raise

    result = try_claim_pending()
    if result:
        return result

    # No pending batches, check if we need to create more
    cursor.execute("SELECT COALESCE(MAX(end_idx), 0) FROM work_queue")
    last_end = cursor.fetchone()[0]

    if last_end >= total_size:
        return None  # All work assigned

    # Create new batch
    start_idx = last_end
    end_idx = min(start_idx + batch_size, total_size)

    @retry_on_locked
    def create_new_batch():
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # Re-check the max end_idx inside the transaction
            cursor.execute("SELECT COALESCE(MAX(end_idx), 0) FROM work_queue")
            current_last_end = cursor.fetchone()[0]
            
            if current_last_end >= total_size:
                cursor.execute("ROLLBACK")
                return None  # All work already assigned
                
            # Update start/end based on current state
            start_idx = current_last_end
            end_idx = min(start_idx + batch_size, total_size)
            
            cursor.execute("""
                INSERT INTO work_queue (start_idx, end_idx, worker_id, status, claimed_at)
                VALUES (?, ?, ?, 'in_progress', ?)
                RETURNING batch_id, start_idx, end_idx
            """, (start_idx, end_idx, worker_id, datetime.now()))

            result = cursor.fetchone()
            cursor.execute("COMMIT")
            if result:
                print(f"    ✓ Worker {worker_id} created and claimed batch {result[0]} (rows {result[1]}-{result[2]})")
            return result
        except sqlite3.IntegrityError:
            # Another worker created this batch, try again
            cursor.execute("ROLLBACK")
            return None
        except Exception:
            cursor.execute("ROLLBACK")
            raise

    result = create_new_batch()
    if result:
        return result
    else:
        # Another worker created this batch, try claiming again
        return claim_batch(cursor, conn, worker_id, batch_size, total_size)


def mark_batch_complete(cursor, conn, batch_id):
    """Mark a batch as completed"""
    @retry_on_locked
    def update_batch():
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute("""
                UPDATE work_queue 
                SET status = 'completed', completed_at = ?
                WHERE batch_id = ?
            """, (datetime.now(), batch_id))
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    update_batch()


def merge_labels(example):
    """Merge onderwerp and beleving labels from the UWV dataset format"""
    all_labels = []
    
    # Process onderwerp labels - extract sub-signal part after " - "
    if example.get('gpt41_onderwerp_labels'):
        for label in example['gpt41_onderwerp_labels']:
            if ' - ' in label:
                # Extract the sub-signal part (after the dash)
                sub_label = label.split(' - ', 1)[1]
                all_labels.append(sub_label)
            else:
                all_labels.append(label)
    
    # Process beleving labels - extract sub-signal part after " - "
    if example.get('gpt41_beleving_labels'):
        for label in example['gpt41_beleving_labels']:
            if ' - ' in label:
                # Extract the sub-signal part (after the dash)
                sub_label = label.split(' - ', 1)[1]
                all_labels.append(sub_label)
            else:
                all_labels.append(label)
    
    return {
        'text': example['text'],
        'gold_labels': str(all_labels),  # Convert to string format
        'unique_id': str(example['signal_id']),
        'category': str(example.get('channel', '')),  # Convert channel to string for category
    }


def process_item(item, pipeline, db_path, worker_id):
    """Process a single item through the pipeline"""
    # Convert string unique_id to a stable integer hash for wiki_id
    # This is needed because LoggingLLMWrapper expects source_id to be an integer
    # Use a deterministic hash that's consistent across runs
    unique_id = item["unique_id"]
    import hashlib
    wiki_id = int(hashlib.md5(unique_id.encode()).hexdigest()[:8], 16)  # First 8 hex chars as int
    
    state = TextToKGState(
        text=item["text"],
        category=item.get("category", ""),
        db_path=db_path,
        worker_id=worker_id,
        wiki_id=wiki_id  # Using hashed unique_id as wiki_id
    )

    # Run pipeline
    state = pipeline.invoke(state)
    print("    ✓ Finished processing")

    return state


def main():
    """Main entry point for parallel processing"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Parallel distillation for all graph nodes with automatic retry"
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        help="Worker identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Records per batch (default: 20)"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum records to process (default: all)"
    )
    parser.add_argument(
        "--add-labels", action="store_true", help="Add labels using AddLabelsNode (optional step)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GPT4O",
        choices=["GPT4O", "O3_MINI", "O4_MINI", "GPT4O_MINI", "GPT41"],
        help="Model to use for all nodes (default: GPT4O)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help='JSON config for per-node models, e.g. \'{"n1": "GPT4O", "n2": "O3_MINI", "n3": "O3_MINI"}\'',
    )
    parser.add_argument(
        "--model-preset",
        type=str,
        choices=["high_quality", "balanced", "cost_effective"],
        help="Use a predefined model configuration preset",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./src/data/distill_graph.db",
        help="Path to the SQLite database (default: ./src/data/distill_graph.db)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UWV/wim-synthetic-data-rd",
        help="HuggingFace dataset to use (default: UWV/wim-synthetic-data-rd)"
    )
    parser.add_argument(
        "--clean-queue",
        action="store_true",
        help="Clear the entire work queue for a fresh start (only affects worker0)"
    )
    args = parser.parse_args()

    # Check if taxonomy Excel file exists (required for n5_add_labels_node)
    if args.add_labels:
        taxonomy_file = Path("src/data/Hoofdklantsignalen - Subklantsignalen.xlsx")
        if not taxonomy_file.exists():
            print(f"\n❌ CRITICAL ERROR: Taxonomy file not found at {taxonomy_file}")
            print("This file is required for the label addition step.")
            print("Please ensure the file exists before running the script.")
            sys.exit(1)

    # Generate worker ID if not provided
    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    print(f"Worker ID: {worker_id}")
    print(f"Batch size: {args.batch_size}")
    
    # Add staggered start delay for workers to avoid database lock conflicts
    if args.worker_id and args.worker_id.startswith("worker"):
        try:
            worker_num = int(args.worker_id.replace("worker", ""))
            delay = worker_num * 2  # 2 seconds per worker number
            if delay > 0:
                print(f"Waiting {delay} seconds before starting (staggered start)...")
                time.sleep(delay)
        except ValueError:
            pass  # If worker_id doesn't follow pattern, skip delay

    # Determine model configuration
    if args.model_preset:
        # Load preset from models.yaml
        config_path = os.path.join(os.path.dirname(__file__), "../config/models.yaml")
        with open(config_path, 'r') as f:
            models_config = yaml.safe_load(f)
        
        preset = models_config['cost_optimization'][args.model_preset]
        model_config = {
            'n1': preset['n1'],
            'n2': preset['n2'],
            'n3': preset['n3']
        }
        if args.add_labels:
            model_config['n5'] = preset.get('n5', preset['n3'])  # Default n5 to n3 if not specified
        print(f"Using preset '{args.model_preset}': {model_config}")
    elif args.model_config:
        # Parse JSON config
        model_config = json.loads(args.model_config)
        print(f"Using custom model config: {model_config}")
    else:
        # Use single model for all nodes
        model_config = None
        print(f"Using single model for all nodes: {args.model}")

    # Initialize the pipeline with appropriate configuration
    if model_config:
        # Create LLM instances for each node
        llms = {}
        for node, model_name in model_config.items():
            llms[node] = azure_llm(model_prefix=model_name, temperature=0.0)
        pipeline = TextToKGPipeline(llm=llms, add_labels=args.add_labels).compile()
    else:
        # Single model for all nodes
        llm = azure_llm(model_prefix=args.model, temperature=0.0)
        pipeline = TextToKGPipeline(llm=llm, add_labels=args.add_labels).compile()

    # Initialize db connection with WAL mode for better concurrency
    # Retry connection and initialization to handle concurrent worker starts
    @retry_on_locked
    def init_database():
        # Connect with timeout (built-in SQLite timeout, separate from busy_timeout)
        conn = sqlite3.connect(args.db_path, timeout=60.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Essential PRAGMA settings for concurrent access
        pragmas = [
            "PRAGMA journal_mode=WAL",          # Enable Write-Ahead Logging
            "PRAGMA busy_timeout=60000",        # 60 second busy timeout
            "PRAGMA synchronous=NORMAL",        # Safe with WAL, much faster
            "PRAGMA cache_size=-64000",         # 64MB cache (negative = KiB)
            "PRAGMA temp_store=MEMORY",         # Store temp tables in memory
            "PRAGMA mmap_size=268435456",       # 256MB memory-mapped I/O
            "PRAGMA wal_autocheckpoint=1000",   # Auto-checkpoint every 1000 pages
            "PRAGMA journal_size_limit=67108864", # 64MB WAL file limit
        ]
        
        for pragma in pragmas:
            cursor.execute(pragma)
        
        # Verify WAL mode was enabled
        result = cursor.execute("PRAGMA journal_mode").fetchone()
        if result[0] != 'wal':
            print(f"Warning: Failed to enable WAL mode, got: {result[0]}")
        
        # Initialize tables using the existing connection
        init_tables(conn, cursor, add_labels=args.add_labels)
        return conn, cursor
    
    conn, cursor = init_database()

    # Clean up stale work queue entries
    print("\nCleaning up work queue...")
    @retry_on_locked
    def cleanup_work_queue():
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # If this is the first worker (worker0 or no worker_id), handle queue cleanup
            if not args.worker_id or args.worker_id == "worker0":
                # Reset ALL in-progress batches to pending (since we're restarting all workers)
                cursor.execute("""
                    UPDATE work_queue 
                    SET status = 'pending', worker_id = NULL, claimed_at = NULL
                    WHERE status = 'in_progress'
                """)
                reset_count = cursor.rowcount
                
                if reset_count > 0:
                    print(f"✓ Reset {reset_count} in-progress batches to pending (fresh restart)")
                
                # Check if we should clear the queue
                cursor.execute("SELECT COUNT(*) FROM work_queue")
                total_batches = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM work_queue WHERE status = 'completed'")
                completed_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM processing_status WHERE processing_complete = TRUE")
                processed_count = cursor.fetchone()[0]
                
                # Only clear if explicitly requested OR if there's a mismatch
                if args.clean_queue:
                    cursor.execute("DELETE FROM work_queue")
                    print("✓ Cleared entire work queue (--clean-queue flag)")
                elif total_batches == 0:
                    # No work queue exists yet, this is fine
                    print("✓ No existing work queue found")
                elif completed_count == total_batches and processed_count == 0:
                    # All batches marked complete but no actual records - clear it
                    cursor.execute("DELETE FROM work_queue")
                    print("✓ Cleared work queue (all batches completed but no processed records)")
                else:
                    # Keep the existing queue
                    print(f"✓ Keeping existing work queue ({total_batches} batches: "
                          f"{completed_count} completed, {processed_count} records processed)")
            else:
                # For other workers, don't touch the work queue at all during initial startup
                print("✓ Work queue management skipped (not worker0)")
            
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    
    cleanup_work_queue()

    # Find and delete incomplete records (started but not marked complete)
    print("\nChecking for incomplete pipeline runs...")
    cursor.execute("""
        SELECT DISTINCT unique_id FROM (
            SELECT unique_id FROM n1
            UNION ALL
            SELECT unique_id FROM n2
            UNION ALL
            SELECT unique_id FROM n3
        ) all_ids
        WHERE unique_id NOT IN (
            SELECT unique_id FROM processing_status WHERE processing_complete = TRUE
        )
    """)
    incomplete_ids = [row[0] for row in cursor.fetchall()]
    
    if incomplete_ids:
        print(f"Found {len(incomplete_ids)} incomplete records, cleaning up...")
        # Delete from all tables
        tables_to_clean = ['n1', 'n2', 'n3', 'processing_status']
        if args.add_labels:
            tables_to_clean.append('n5')
        
        for table in tables_to_clean:
            # Check if n5 table exists before trying to delete
            if table == 'n5':
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='n5'")
                if not cursor.fetchone():
                    continue
            cursor.execute(f"DELETE FROM {table} WHERE unique_id IN ({','.join('?' * len(incomplete_ids))})", incomplete_ids)
        print(f"✓ Deleted {len(incomplete_ids)} incomplete records")

    # Get detailed progress information
    print("\nChecking processing progress...")
    
    # Get counts from each stage
    cursor.execute("SELECT COUNT(DISTINCT unique_id) FROM n1")
    n1_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT unique_id) FROM n2")
    n2_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT unique_id) FROM n3")
    n3_count = cursor.fetchone()[0]
    
    if args.add_labels:
        # Check if n5 table exists first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='n5'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(DISTINCT unique_id) FROM n5")
            n5_count = cursor.fetchone()[0]
        else:
            n5_count = 0
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN processing_complete = TRUE THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN processing_complete = TRUE AND labels_added = TRUE THEN 1 ELSE 0 END) as with_labels
        FROM processing_status
    """)
    status_info = cursor.fetchone()
    
    print(f"\nPipeline Progress Summary:")
    print(f"  • Stage n1 (Entity Extraction): {n1_count} records")
    print(f"  • Stage n2 (Schema Mapping): {n2_count} records")
    print(f"  • Stage n3 (JSON-LD Generation): {n3_count} records")
    if args.add_labels:
        print(f"  • Stage n5 (Label Addition): {n5_count} records")
    print(f"  • Fully completed: {status_info['completed'] if status_info else 0} records")
    if args.add_labels:
        print(f"  • Completed with labels: {status_info['with_labels'] if status_info else 0} records")
    
    # Get work queue status
    cursor.execute("""
        SELECT 
            COUNT(*) as total_batches,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
        FROM work_queue
    """)
    queue_info = cursor.fetchone()
    if queue_info and queue_info['total_batches'] > 0:
        print(f"\nWork Queue Status:")
        print(f"  • Total batches: {queue_info['total_batches']}")
        print(f"  • Pending: {queue_info['pending']}")
        print(f"  • In progress: {queue_info['in_progress']}")
        print(f"  • Completed: {queue_info['completed']}")

    # Get already completed unique_ids
    if args.add_labels:
        # When adding labels, exclude only records that are complete AND already have labels
        cursor.execute("""
            SELECT unique_id FROM processing_status 
            WHERE processing_complete = TRUE AND labels_added = TRUE
        """)
        completed_ids = set(row[0] for row in cursor.fetchall())
        print(f"\nFiltering out {len(completed_ids)} fully completed records with labels")
    else:
        # Without labels, exclude all completed records
        cursor.execute("""
            SELECT unique_id FROM processing_status 
            WHERE processing_complete = TRUE
        """)
        completed_ids = set(row[0] for row in cursor.fetchall())
        print(f"\nFiltering out {len(completed_ids)} completed records")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    raw_dataset = load_dataset(args.dataset, split="train")
    
    # Map the dataset to merge labels
    dataset = raw_dataset.map(merge_labels, remove_columns=raw_dataset.column_names)
    
    # Filter out completed items
    dataset = dataset.filter(lambda x: x["unique_id"] not in completed_ids)
    
    # Apply limit if specified
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    total_size = len(dataset)
    print(f"Dataset loaded: {total_size} records to process")

    if total_size == 0:
        print("No records to process!")
        return

    # Process batches until none left
    batches_processed = 0
    
    # Show current work queue state before processing
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
               SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
        FROM work_queue
    """)
    queue_state = cursor.fetchone()
    print(f"\nWork queue before processing: Total={queue_state['total']}, Pending={queue_state['pending']}, "
          f"InProgress={queue_state['in_progress']}, Completed={queue_state['completed']}")
    
    while True:
        # Claim a batch
        batch_info = claim_batch(cursor, conn, worker_id, args.batch_size, total_size, args.db_path)
        if not batch_info:
            print(f"\n✓ Worker {worker_id}: No more batches to process")
            break

        batch_id, start_idx, end_idx = batch_info
        batch_size = end_idx - start_idx
        batches_processed += 1
        print(f"\n{'='*60}")
        print(f"Worker {worker_id} processing batch {batch_id}: records {start_idx}-{end_idx} ({batch_size} items)")
        print(f"{'='*60}")

        # Get the subset of data for this batch
        batch_dataset = dataset.select(range(start_idx, end_idx))

        # Process each item in the batch
        batch_start_time = time.time()
        items_processed = 0
        items_skipped = 0

        for i, item in enumerate(batch_dataset):
            unique_id = item["unique_id"]
            # Generate stable wiki_id from unique_id string (same as in process_item)
            import hashlib
            wiki_id = int(hashlib.md5(unique_id.encode()).hexdigest()[:8], 16)
            print(f"\n  [{i+1}/{len(batch_dataset)}] Processing unique_id: {unique_id} (wiki_id: {wiki_id})")

            # Check if already processed (for resumption)
            @retry_on_locked
            def check_already_processed():
                if args.add_labels:
                    # In add-labels mode, check if record is complete with labels
                    cursor.execute("""
                        SELECT processing_complete, labels_added 
                        FROM processing_status 
                        WHERE unique_id = ?
                    """, (unique_id,))
                    result = cursor.fetchone()
                    # Skip if fully complete with labels, process if needs labels
                    return result and result[0] and result[1]
                else:
                    # Without labels, check if record is marked complete
                    cursor.execute("""
                        SELECT processing_complete 
                        FROM processing_status 
                        WHERE unique_id = ?
                    """, (unique_id,))
                    result = cursor.fetchone()
                    return result and result[0]

            if check_already_processed():
                print("    → Already processed, skipping")
                items_skipped += 1
                continue

            try:
                # Process through pipeline
                state = process_item(item, pipeline, args.db_path, worker_id)

                # Add data to all required tables in a single transaction
                @retry_on_locked
                def insert_all_records():
                    # Explicit transaction for all inserts
                    cursor.execute("BEGIN IMMEDIATE")
                    try:
                        cursor.execute(
                            "INSERT INTO n1 (unique_id, system_prompt, user_prompt, input_text, output_text) VALUES (?, ?, ?, ?, ?)",
                            (
                                unique_id,  # Keep original string ID
                                ENTITY_EXTRACTION_SYSTEM_PROMPT,
                                ENTITY_EXTRACTION_HUMAN_PROMPT,
                                item["text"],
                                (
                                    str(state["entity_extraction_output"])
                                    if state["entity_extraction_output"]
                                    else None
                                ),
                            ),
                        )
                        cursor.execute(
                            "INSERT INTO n2 (unique_id, system_prompt, user_prompt, input_text, output_text) VALUES (?, ?, ?, ?, ?)",
                            (
                                unique_id,  # Keep original string ID
                                RETRIEVE_SCHEMA_ORG_SYSTEM_PROMPT,
                                RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT,
                                item["text"],
                                str(state["schema_definitions"]),
                            ),
                        )
                        cursor.execute(
                            "INSERT INTO n3 (unique_id, system_prompt, user_prompt, input_text, output_text, err_msg_1, err_msg_2, err_msg_3, err_msg_4, err_msg_5, err_out_1, err_out_2, err_out_3, err_out_4, err_out_5, validation_failed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                unique_id,  # Keep original string ID
                                TRANSFORM_TO_KG_SYSTEM_PROMPT,
                                TRANSFORM_TO_KG_HUMAN_PROMPT,
                                state["text"],
                                (
                                    state["json_ld_contents"][-1]
                                    if state.get("json_ld_contents")
                                    else None
                                ),
                                (
                                    state["validation_output"][0]
                                    if state["validation_output"]
                                    else None
                                ),
                                (
                                    state["validation_output"][1]
                                    if len(state["validation_output"]) > 1
                                    else None
                                ),
                                (
                                    state["validation_output"][2]
                                    if len(state["validation_output"]) > 2
                                    else None
                                ),
                                (
                                    state["validation_output"][3]
                                    if len(state["validation_output"]) > 3
                                    else None
                                ),
                                (
                                    state["validation_output"][4]
                                    if len(state["validation_output"]) > 4
                                    else None
                                ),
                                (
                                    state["json_ld_contents"][0]
                                    if state.get("json_ld_contents")
                                    else None
                                ),
                                (
                                    state["json_ld_contents"][1]
                                    if len(state.get("json_ld_contents", [])) > 1
                                    else None
                                ),
                                (
                                    state["json_ld_contents"][2]
                                    if len(state.get("json_ld_contents", [])) > 2
                                    else None
                                ),
                                (
                                    state["json_ld_contents"][3]
                                    if len(state.get("json_ld_contents", [])) > 3
                                    else None
                                ),
                                (
                                    state["json_ld_contents"][4]
                                    if len(state.get("json_ld_contents", [])) > 4
                                    else None
                                ),
                                state["validation_max_runs_reached"],
                            ),
                        )
                        
                        # Add n5 record if labels are enabled
                        if args.add_labels:
                            # Extract labels from the final JSON-LD
                            labels_list = []
                            if state.get("json_ld_contents") and len(state["json_ld_contents"]) > 0:
                                try:
                                    # The AddLabelsNode modifies the last json_ld_contents entry
                                    final_json_ld = json.loads(state["json_ld_contents"][-1])
                                    if isinstance(final_json_ld, dict) and "labels" in final_json_ld:
                                        labels_list = final_json_ld["labels"]
                                except (json.JSONDecodeError, KeyError):
                                    pass
                            
                            cursor.execute(
                                "INSERT INTO n5 (unique_id, system_prompt, user_prompt, input_text, output_text, labels) VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    unique_id,  # Keep original string ID
                                    ADD_LABELS_SYSTEM_PROMPT,
                                    ADD_LABELS_HUMAN_PROMPT,
                                    state["text"],
                                    (
                                        state["json_ld_contents"][-1]
                                        if state.get("json_ld_contents")
                                        else None
                                    ),
                                    json.dumps(labels_list) if labels_list else None,
                                ),
                            )
                        
                        # Update processing status
                        cursor.execute(
                            """INSERT OR REPLACE INTO processing_status 
                               (unique_id, processing_complete, labels_added, processed_at, worker_id) 
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                unique_id,  # Keep original string ID
                                True,
                                args.add_labels,
                                datetime.now().isoformat(),
                                worker_id,
                            ),
                        )
                        
                        # Commit the transaction
                        cursor.execute("COMMIT")
                    except Exception:
                        cursor.execute("ROLLBACK")
                        raise

                insert_all_records()
                items_processed += 1

                # Clean up tmp folder AFTER transaction is committed
                if os.path.exists("src/data/tmp"):
                    shutil.rmtree("src/data/tmp")

            except BadRequestError as e:
                if "content_filter" in str(e):
                    print("    ⚠ Skipping due to Azure OpenAI content filter")
                elif "context_length" in str(e):
                    print("    ⚠ Skipping due to context length")
                else:
                    print(f"    ❌ BadRequestError processing item: {e}")
                    raise e
                items_skipped += 1
                continue
            except ValueError as e:
                if "content filter" in str(e):
                    print("    ⚠ Skipping due to Azure OpenAI content filter (ValueError)")
                else:
                    print(f"    ❌ ValueError processing item: {e}")
                items_skipped += 1
                continue
            except json.JSONDecodeError as e:
                print(f"    ⚠ Skipping due to JSONDecodeError: {e}")
                items_skipped += 1
                continue
            except Exception as e:
                # Check for critical errors that should stop processing
                error_str = str(e)
                if "404" in error_str and "Resource not found" in error_str:
                    print(f"\n❌ CRITICAL ERROR: {e}")
                    print("The Azure OpenAI deployment was not found. Please check your .env configuration:")
                    print(f"  - Model: {args.model}")
                    print(f"  - Deployment name: {args.model.lower().replace('_', '-')}")
                    print("\nExiting...")
                    conn.close()
                    sys.exit(1)
                
                print(f"    ❌ Error processing item: {e}")
                items_skipped += 1
                continue

        # Mark batch as complete
        mark_batch_complete(cursor, conn, batch_id)

        # Print batch summary
        batch_time = time.time() - batch_start_time
        rate = items_processed / batch_time if batch_time > 0 else 0
        print(f"\n{'='*60}")
        print(f"Batch {batch_id} completed in {batch_time:.1f}s")
        print(f"  • Processed: {items_processed}")
        print(f"  • Skipped: {items_skipped}")
        print(f"  • Rate: {rate:.2f} items/s")
        print(f"{'='*60}")

    # Run PRAGMA optimize before closing
    try:
        cursor.execute("PRAGMA optimize")
        print("✓ Optimized database statistics")
    except:
        pass  # Ignore optimize errors
    
    conn.close()
    print(f"\n✓ Worker {worker_id} finished - processed {batches_processed} batches")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Check the progress of the distillation pipeline by querying the SQLite database.
"""

import sqlite3
import argparse
from datetime import datetime
import pandas as pd


def check_progress(db_path):
    """Check and display the progress of the distillation pipeline"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print(f"\n{'='*60}")
    print(f"Distillation Progress Report")
    print(f"Database: {db_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    if not any(table in ['n1', 'n2', 'n3', 'processing_status'] for table in tables):
        print("\n❌ No distillation tables found in database!")
        print("The database might be empty or not initialized.")
        return
    
    # Get counts from each stage
    stages = {
        'n1': 'Entity Extraction',
        'n2': 'Schema Mapping', 
        'n3': 'JSON-LD Generation',
        'n5': 'Label Addition'
    }
    
    print("\n📊 Pipeline Stage Progress:")
    for table, description in stages.items():
        if table in tables:
            cursor.execute(f"SELECT COUNT(DISTINCT unique_id) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  • {description} ({table}): {count:,} records")
    
    # Get detailed status information
    if 'processing_status' in tables:
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN processing_complete = TRUE THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN processing_complete = TRUE AND labels_added = TRUE THEN 1 ELSE 0 END) as with_labels
            FROM processing_status
        """)
        status = cursor.fetchone()
        
        print(f"\n✅ Completion Status:")
        print(f"  • Total tracked: {status['total']:,}")
        print(f"  • Fully completed: {status['completed']:,}")
        print(f"  • With labels: {status['with_labels']:,}")
    
    # Check validation failures
    if 'n3' in tables:
        cursor.execute("""
            SELECT COUNT(*) FROM n3 WHERE validation_failed = TRUE
        """)
        failed_count = cursor.fetchone()[0]
        if failed_count > 0:
            print(f"\n⚠️  Validation failures: {failed_count:,} records")
    
    # Work queue status
    if 'work_queue' in tables:
        cursor.execute("""
            SELECT 
                COUNT(*) as total_batches,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                MIN(start_idx) as first_idx,
                MAX(end_idx) as last_idx
            FROM work_queue
        """)
        queue = cursor.fetchone()
        
        if queue['total_batches'] > 0:
            print(f"\n📋 Work Queue Status:")
            print(f"  • Total batches: {queue['total_batches']:,}")
            print(f"  • Pending: {queue['pending']:,}")
            print(f"  • In progress: {queue['in_progress']:,}")
            print(f"  • Completed: {queue['completed']:,}")
            print(f"  • Index range: {queue['first_idx']:,} - {queue['last_idx']:,}")
            
            # Show active workers
            cursor.execute("""
                SELECT worker_id, COUNT(*) as batch_count, MAX(claimed_at) as last_claim
                FROM work_queue
                WHERE status = 'in_progress'
                GROUP BY worker_id
            """)
            active_workers = cursor.fetchall()
            if active_workers:
                print(f"\n👷 Active Workers:")
                for worker in active_workers:
                    print(f"  • {worker['worker_id']}: {worker['batch_count']} batch(es), last claim: {worker['last_claim']}")
    
    # Recent activity
    if 'processing_status' in tables:
        cursor.execute("""
            SELECT worker_id, COUNT(*) as records_processed, MAX(processed_at) as last_activity
            FROM processing_status
            WHERE processed_at > datetime('now', '-1 hour')
            GROUP BY worker_id
            ORDER BY last_activity DESC
            LIMIT 10
        """)
        recent_activity = cursor.fetchall()
        
        if recent_activity:
            print(f"\n🕐 Recent Activity (last hour):")
            for activity in recent_activity:
                print(f"  • {activity['worker_id']}: {activity['records_processed']} records, last: {activity['last_activity']}")
    
    # Sample of recent records
    print(f"\n📝 Sample of Recent Records:")
    cursor.execute("""
        SELECT ps.unique_id, ps.processing_complete, ps.labels_added, ps.processed_at, ps.worker_id
        FROM processing_status ps
        ORDER BY ps.processed_at DESC
        LIMIT 5
    """)
    recent_records = cursor.fetchall()
    
    if recent_records:
        for rec in recent_records:
            status_str = "✅ Complete" if rec['processing_complete'] else "⏳ In Progress"
            labels_str = " + Labels" if rec['labels_added'] else ""
            print(f"  • {rec['unique_id']}: {status_str}{labels_str} by {rec['worker_id']} at {rec['processed_at']}")
    else:
        print("  No records found")
    
    # LLM call statistics
    if 'distill_llm_calls' in tables:
        cursor.execute("""
            SELECT 
                call_name,
                COUNT(*) as call_count,
                COUNT(DISTINCT source_id) as unique_sources,
                COUNT(DISTINCT worker_id) as unique_workers
            FROM distill_llm_calls
            GROUP BY call_name
        """)
        llm_stats = cursor.fetchall()
        
        if llm_stats:
            print(f"\n🤖 LLM Call Statistics:")
            for stat in llm_stats:
                print(f"  • {stat['call_name']}: {stat['call_count']:,} calls, "
                      f"{stat['unique_sources']:,} unique sources, "
                      f"{stat['unique_workers']} workers")
    
    print(f"\n{'='*60}\n")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Check distillation pipeline progress")
    parser.add_argument(
        "--db-path",
        type=str,
        default="src/data/distill_graph.db",
        help="Path to the SQLite database (default: src/data/distill_graph.db)"
    )
    
    args = parser.parse_args()
    check_progress(args.db_path)


if __name__ == "__main__":
    main()
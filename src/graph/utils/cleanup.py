import os
from typing import List


def cleanup_n3_files(state):
    """
    Clean up n3 files after successful n5 file creation.
    
    Args:
t        state: TextToKGState or dict containing file paths
    """
    # Handle both TextToKGState objects and dict results
    json_ld_paths = state.json_ld_paths if hasattr(state, 'json_ld_paths') else state.get('json_ld_paths', [])
    n5_file_paths = state.n5_file_paths if hasattr(state, 'n5_file_paths') else state.get('n5_file_paths', [])
    
    # Only cleanup if we have both n3 and n5 files
    if not json_ld_paths or not n5_file_paths:
        return
    
    # For each n5 file that was successfully created, delete the corresponding n3 file
    for n5_path in n5_file_paths:
        # Extract UUID from n5 path
        uuid_part = n5_path.split('n5_')[-1].replace('.json', '')
        n3_path = f"src/data/tmp/n3_{uuid_part}.json"
        
        # Check if the n3 file exists and delete it
        if os.path.exists(n3_path):
            try:
                os.remove(n3_path)
                print(f"    → Cleaned up intermediate file: {n3_path}")
            except Exception as e:
                print(f"    → Warning: Could not delete {n3_path}: {e}")


def cleanup_old_tmp_files(directory: str = "src/data/tmp", max_age_hours: int = 24):
    """
    Clean up old temporary files based on age.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours before deletion
    """
    import time
    
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        if filename.startswith(('n3_', 'n5_')) and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    print(f"    → Cleaned up old file: {filename} (age: {file_age/3600:.1f} hours)")
            except Exception as e:
                print(f"    → Warning: Could not process {filename}: {e}")
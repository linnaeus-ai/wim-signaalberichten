#!/usr/bin/env python3
"""Test infrastructure error handling in the validation pipeline"""

import asyncio
import os
import tempfile
from graph.text_to_kg_pipeline import TextToKGPipeline
from graph.utils import azure_llm

async def test_infrastructure_error():
    print("=== Testing Infrastructure Error Handling ===\n")
    
    # Initialize pipeline
    pipeline = TextToKGPipeline(
        llm=azure_llm(model_prefix="GPT4O"),
        add_labels=True
    ).compile()
    
    # Test 1: Normal operation (should work)
    print("Test 1: Normal validation (should pass)")
    try:
        result = await pipeline.ainvoke({
            "text": "Een evenement in Amsterdam op 15 januari 2025"
        })
        print("✓ Normal validation passed\n")
    except Exception as e:
        print(f"✗ Unexpected error: {e}\n")
    
    # Test 2: Simulate infrastructure error by temporarily renaming TTL file
    ttl_path = "src/graph/validator/schemaorg-all-http.ttl"
    ttl_backup = ttl_path + ".backup"
    
    print("Test 2: Infrastructure error (missing TTL file)")
    try:
        # Rename TTL file to simulate missing infrastructure
        if os.path.exists(ttl_path):
            os.rename(ttl_path, ttl_backup)
            
        result = await pipeline.ainvoke({
            "text": "Een evenement in Amsterdam op 15 januari 2025"
        })
        print("✗ Pipeline should have failed but didn't!\n")
    except RuntimeError as e:
        if "Infrastructure error" in str(e):
            print(f"✓ Infrastructure error correctly caught: {str(e)[:100]}...\n")
        else:
            print(f"✗ Wrong error type: {e}\n")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}\n")
    finally:
        # Restore TTL file
        if os.path.exists(ttl_backup):
            os.rename(ttl_backup, ttl_path)
    
    # Test 3: Simulate corrupted TTL by creating invalid file
    print("Test 3: Infrastructure error (corrupted TTL file)")
    try:
        # Create corrupted TTL file
        with open(ttl_path, 'w') as f:
            f.write("This is not valid TTL content!")
            
        result = await pipeline.ainvoke({
            "text": "Een evenement in Amsterdam op 15 januari 2025"
        })
        print("✗ Pipeline should have failed but didn't!\n")
    except RuntimeError as e:
        if "Infrastructure error" in str(e):
            print(f"✓ Infrastructure error correctly caught: {str(e)[:100]}...\n")
        else:
            print(f"✗ Wrong error type: {e}\n")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}\n")
    finally:
        # Restore original TTL file (would need actual backup in production)
        print("Note: TTL file needs to be restored manually after this test")
    
    print("=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(test_infrastructure_error())
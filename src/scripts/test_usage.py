#!/usr/bin/env python3
"""Test the usage example from README.md"""

import asyncio
import json
from graph.text_to_kg_pipeline import TextToKGPipeline
from graph.utils import azure_llm

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

async def test_usage():
    # Input text
    input_text = "Mark Rutte (Den Haag, 14 februari 1967) was van 14 oktober 2010 tot 2 juli 2024 minister-president van Nederland"
    
    print(f"{Colors.HEADER}{Colors.BOLD}=== Text-to-Knowledge-Graph Pipeline Test ==={Colors.ENDC}\n")
    print(f"{Colors.CYAN}Input text:{Colors.ENDC} \"{input_text}\"\n")
    
    # Initialize pipeline with Azure LLM
    print(f"{Colors.YELLOW}Initializing pipeline...{Colors.ENDC}")
    pipeline = TextToKGPipeline(
        llm=azure_llm(model_prefix="GPT4O"),  # Using default GPT4O model
        add_labels=True
    ).compile()
    print(f"{Colors.GREEN}✓ Pipeline initialized{Colors.ENDC}\n")
    
    # Process text
    print(f"{Colors.YELLOW}Processing text through pipeline...{Colors.ENDC}")
    result = await pipeline.ainvoke(
        {"text": input_text},
        config={"configurable": {"max_retries": 3}}
    )
    print(f"{Colors.GREEN}✓ Pipeline execution completed!{Colors.ENDC}\n")
    
    # Print entity extraction results
    if "entity_extraction_output" in result:
        print(f"{Colors.HEADER}{Colors.BOLD}Entity Extraction Results:{Colors.ENDC}")
        extraction = result["entity_extraction_output"]
        
        # Print summary
        if "summary" in extraction:
            print(f"  {Colors.YELLOW}Summary:{Colors.ENDC} {extraction['summary']}")
            print()
        
        # Print entities
        if "entities" in extraction and extraction["entities"]:
            print(f"  {Colors.YELLOW}Entities:{Colors.ENDC}")
            for entity_triple in extraction["entities"]:
                if len(entity_triple) >= 3:
                    entity_name, class_name, description = entity_triple
                    print(f"    • {Colors.CYAN}{entity_name}{Colors.ENDC} ({Colors.GREEN}{class_name}{Colors.ENDC}): {description}")
        
        # Print relations
        if "relations" in extraction and extraction["relations"]:
            print(f"\n  {Colors.YELLOW}Relations:{Colors.ENDC}")
            for relation in extraction["relations"]:
                if len(relation) >= 3:
                    subject, predicate, obj = relation
                    print(f"    • {Colors.CYAN}{subject}{Colors.ENDC} → {Colors.YELLOW}{predicate}{Colors.ENDC} → {Colors.CYAN}{obj}{Colors.ENDC}")
        print()
    
    # Print schema mappings
    if "schema_definitions" in result and result["schema_definitions"]:
        print(f"{Colors.HEADER}{Colors.BOLD}Schema Mappings:{Colors.ENDC}")
        schema_defs = result["schema_definitions"]
        
        # schema_definitions is a Dict[str, str] where key is class name and value is YAML definition
        if isinstance(schema_defs, dict):
            for class_name, yaml_def in schema_defs.items():
                # Extract Schema.org type from YAML definition
                # The YAML starts with the schema type followed by a colon
                if yaml_def and isinstance(yaml_def, str):
                    lines = yaml_def.strip().split('\n')
                    if lines:
                        schema_type = lines[0].rstrip(':')
                        print(f"  • {Colors.CYAN}{class_name}{Colors.ENDC} → {Colors.GREEN}{schema_type}{Colors.ENDC}")
        print()
    
    # Print validation results
    if "validation_runs" in result:
        validation_status = "PASSED" if result.get("validation_returncode") == 0 else "FAILED"
        color = Colors.GREEN if validation_status == "PASSED" else Colors.RED
        print(f"{Colors.HEADER}{Colors.BOLD}Validation Results:{Colors.ENDC}")
        print(f"  Status: {color}{validation_status}{Colors.ENDC}")
        print(f"  Attempts: {result['validation_runs']}/{result['validation_max_runs']}")
        if result.get("validation_output"):
            print(f"  {Colors.YELLOW}Last error:{Colors.ENDC} {result['validation_output'][:100]}...")
        print()
    
    # Print JSON-LD output
    if "json_ld_contents" in result and result["json_ld_contents"]:
        print(f"{Colors.HEADER}{Colors.BOLD}Generated JSON-LD:{Colors.ENDC}")
        json_ld = json.loads(result["json_ld_contents"][-1])
        formatted_json = json.dumps(json_ld, indent=2, ensure_ascii=False)
        # Add syntax highlighting to JSON
        formatted_json = formatted_json.replace('"@context":', f'{Colors.YELLOW}"@context":{Colors.ENDC}')
        formatted_json = formatted_json.replace('"@type":', f'{Colors.YELLOW}"@type":{Colors.ENDC}')
        for line in formatted_json.split('\n'):
            if ':' in line and not line.strip().startswith('{'):
                parts = line.split(':', 1)
                key_part = parts[0]
                value_part = parts[1] if len(parts) > 1 else ''
                # Color the keys
                if '"' in key_part:
                    key_start = key_part.find('"')
                    key_end = key_part.rfind('"')
                    colored_key = key_part[:key_start] + f'{Colors.CYAN}{key_part[key_start:key_end+1]}{Colors.ENDC}' + key_part[key_end+1:]
                    line = colored_key + ':' + value_part
            print(line)
        print()
    
    # Print labels if any
    if "labels" in result and result["labels"]:
        print(f"{Colors.HEADER}{Colors.BOLD}Generated Labels:{Colors.ENDC}")
        for label in result["labels"]:
            print(f"  • {Colors.GREEN}{label}{Colors.ENDC}")
        print()
    
    # Summary
    print(f"{Colors.HEADER}{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  • Result keys: {Colors.CYAN}{', '.join(result.keys())}{Colors.ENDC}")
    print(f"  • Total processing nodes: {Colors.GREEN}5{Colors.ENDC} (Entity Extraction → Schema Mapping → KG Generation → Validation → Labeling)")
    
if __name__ == "__main__":
    asyncio.run(test_usage())
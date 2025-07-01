import yaml
import sqlite3
import re


def shorten_schema_yaml(yaml_content: str) -> str:
    """
    Shorten Schema.org YAML by removing redundant URIs and restructuring.
    
    Args:
        yaml_content: Original YAML with full URI definitions
        
    Returns:
        Shortened YAML with URIs removed and more compact structure
    """
    try:
        data = yaml.safe_load(yaml_content)
        
        # Create shortened structure
        shortened = {
            'label': data.get('label'),
            'subClassOf': data.get('subClassOf', [])
        }
        
        # Add comments if present
        if 'comments' in data:
            shortened['comments'] = data['comments']
        
        # Process properties - convert from list to dict format
        if 'properties' in data:
            props = {}
            for prop in data['properties']:
                prop_name = prop['label']
                prop_data = {}
                
                if 'comments' in prop:
                    prop_data['comments'] = prop['comments']
                if 'ranges' in prop:
                    prop_data['ranges'] = prop['ranges']
                
                # Only add property if it has data
                if prop_data:
                    props[prop_name] = prop_data
            
            if props:
                shortened['properties'] = props
        
        # Return shortened YAML
        return yaml.dump(shortened, sort_keys=False, allow_unicode=True, default_flow_style=False)
        
    except Exception as e:
        print(f"Error shortening YAML: {e}")
        return yaml_content  # Return original if error


def ultra_shorten_schema_yaml(yaml_content: str) -> str:
    """
    Ultra-compact Schema.org YAML by aggressive optimization.
    
    Args:
        yaml_content: Original YAML with full definitions
        
    Returns:
        Ultra-shortened YAML with maximum compression
    """
    try:
        data = yaml.safe_load(yaml_content)
        
        # Start with basic structure
        shortened = {
            'label': data.get('label'),
            'subClassOf': data.get('subClassOf', [])
        }
        
        # Flatten single-item arrays in subClassOf
        if len(shortened['subClassOf']) == 1:
            shortened['subClassOf'] = shortened['subClassOf'][0]
        
        # Handle comments - only keep if meaningful
        if 'comments' in data:
            comments = data['comments']
            if len(comments) == 1:
                # Single comment becomes a string
                comment = comments[0]
                # Skip obvious comments that just repeat the class name
                if not re.match(r'^(A|An|The)\s+' + data.get('label', '').lower(), comment.lower()):
                    shortened['comment'] = comment
            elif len(comments) > 1:
                shortened['comments'] = comments
        
        # Process properties with aggressive optimization
        if 'properties' in data:
            # Categorize properties
            simple_props = {}  # Properties with single range, no comment
            text_props = []    # Properties that only accept Text
            complex_props = {} # Properties needing full definition
            
            for prop in data['properties']:
                prop_name = prop['label']
                prop_ranges = prop.get('ranges', [])
                prop_comments = prop.get('comments', [])
                
                # Check if property name is self-explanatory
                self_explanatory = prop_name in [
                    'name', 'description', 'email', 'telephone', 'url', 
                    'image', 'logo', 'identifier', 'alternateName'
                ]
                
                # Decide how to store this property
                if len(prop_ranges) == 1 and (not prop_comments or self_explanatory):
                    # Single range, no meaningful comment
                    range_value = prop_ranges[0]
                    if range_value == 'Text':
                        text_props.append(prop_name)
                    else:
                        simple_props[prop_name] = range_value
                else:
                    # Complex property needs more detail
                    prop_data = {}
                    
                    # Add comment only if meaningful and not self-explanatory
                    if prop_comments and not self_explanatory:
                        if len(prop_comments) == 1:
                            prop_data['comment'] = prop_comments[0]
                        else:
                            prop_data['comments'] = prop_comments
                    
                    # Add ranges
                    if prop_ranges:
                        if len(prop_ranges) == 1:
                            prop_data['range'] = prop_ranges[0]
                        else:
                            prop_data['ranges'] = prop_ranges
                    
                    if prop_data:  # Only add if there's actual data
                        complex_props[prop_name] = prop_data
            
            # Build properties section
            props_section = {}
            
            # Add text properties as a list if there are many
            if len(text_props) > 5:
                props_section['_text'] = text_props
            else:
                # Add them as simple properties
                for prop in text_props:
                    simple_props[prop] = 'Text'
            
            # Add simple properties
            if simple_props:
                props_section.update(simple_props)
            
            # Add complex properties
            if complex_props:
                props_section.update(complex_props)
            
            if props_section:
                shortened['properties'] = props_section
        
        # Custom YAML formatting for maximum compactness
        return format_compact_yaml(shortened)
        
    except Exception as e:
        print(f"Error ultra-shortening YAML: {e}")
        return yaml_content


def format_compact_yaml(data):
    """Format YAML in ultra-compact style"""
    lines = []
    
    # Basic fields
    lines.append(f"label: {data['label']}")
    
    if 'subClassOf' in data:
        if isinstance(data['subClassOf'], list):
            lines.append(f"subClassOf: {data['subClassOf']}")
        else:
            lines.append(f"subClassOf: {data['subClassOf']}")
    
    if 'comment' in data:
        lines.append(f"comment: {data['comment']}")
    elif 'comments' in data:
        lines.append("comments:")
        for c in data['comments']:
            lines.append(f"  - {c}")
    
    # Properties section
    if 'properties' in data:
        lines.append("properties:")
        props = data['properties']
        
        # Handle special _text array first
        if '_text' in props:
            lines.append(f"  _text: {props['_text']}")
        
        # Simple properties (single line)
        for key, value in sorted(props.items()):
            if key == '_text':
                continue
            if isinstance(value, str):
                lines.append(f"  {key}: {value}")
            elif isinstance(value, dict):
                # Complex property
                lines.append(f"  {key}:")
                if 'comment' in value:
                    lines.append(f"    comment: {value['comment']}")
                if 'comments' in value:
                    lines.append("    comments:")
                    for c in value['comments']:
                        lines.append(f"      - {c}")
                if 'range' in value:
                    lines.append(f"    range: {value['range']}")
                if 'ranges' in value:
                    lines.append(f"    ranges: {value['ranges']}")
    
    return '\n'.join(lines)


def test_shorten_yaml():
    """Test the YAML shortening function with real data from database"""
    
    # Connect to database
    db_path = "src/data/schema_definitions.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get Person schema to test text property optimization
        cursor.execute(
            "SELECT yaml_definition FROM schema_classes WHERE class_label = 'Person' LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("Could not find Person schema in database")
            return
        
        original_yaml = row[0]
        shortened_yaml = shorten_schema_yaml(original_yaml)
        
        # Parse both to check for data loss
        original_data = yaml.safe_load(original_yaml)
        shortened_data = yaml.safe_load(shortened_yaml)
        
        # Print comparison
        print(f"=== TESTING YAML: {original_data.get('label', 'Unknown')} ===")
        print(f"Original size: {len(original_yaml)} characters")
        print(f"Shortened size: {len(shortened_yaml)} characters")
        print(f"Size reduction: {(1 - len(shortened_yaml)/len(original_yaml))*100:.1f}%")
        
        # Check what's preserved
        print("\n=== DATA INTEGRITY CHECK ===")
        print(f"Label preserved: {original_data.get('label') == shortened_data.get('label')}")
        print(f"SubClassOf preserved: {original_data.get('subClassOf') == shortened_data.get('subClassOf')}")
        print(f"Comments preserved: {original_data.get('comments') == shortened_data.get('comments')}")
        
        # Check properties
        if 'properties' in original_data:
            original_props = {p['label']: p for p in original_data['properties']}
            shortened_props = shortened_data.get('properties', {})
            
            print(f"\nOriginal properties count: {len(original_props)}")
            print(f"Shortened properties count: {len(shortened_props)}")
            
            # Check if all properties are present
            missing_props = set(original_props.keys()) - set(shortened_props.keys())
            if missing_props:
                print(f"Missing properties: {missing_props}")
            
            # Check a sample property in detail
            if original_props:
                sample_prop = list(original_props.keys())[0]
                print(f"\nSample property '{sample_prop}' comparison:")
                print(f"Original: {original_props[sample_prop]}")
                print(f"Shortened: {shortened_props.get(sample_prop, 'MISSING')}")
        
        # Check for any unexpected fields in original
        original_keys = set(original_data.keys())
        expected_keys = {'label', 'uri', 'subClassOf', 'comments', 'properties'}
        unexpected_keys = original_keys - expected_keys
        if unexpected_keys:
            print(f"\n⚠️  Unexpected fields in original YAML: {unexpected_keys}")
            for key in unexpected_keys:
                print(f"  {key}: {original_data[key]}")
        
        # Show a snippet of the shortened YAML
        print("\n=== SHORTENED YAML SNIPPET ===")
        print(shortened_yaml[:1000] + "..." if len(shortened_yaml) > 1000 else shortened_yaml)
        
        # Test ultra shortening
        print("\n\n=== TESTING ULTRA SHORTENING ===")
        ultra_shortened_yaml = ultra_shorten_schema_yaml(original_yaml)
        print(f"Ultra-shortened size: {len(ultra_shortened_yaml)} characters")
        print(f"Total reduction: {(1 - len(ultra_shortened_yaml)/len(original_yaml))*100:.1f}%")
        print(f"Additional reduction from shortened: {(1 - len(ultra_shortened_yaml)/len(shortened_yaml))*100:.1f}%")
        
        print("\n=== ULTRA-SHORTENED YAML SNIPPET ===")
        print(ultra_shortened_yaml[:1000] + "..." if len(ultra_shortened_yaml) > 1000 else ultra_shortened_yaml)
        
    except Exception as e:
        print(f"Database error: {e}")


if __name__ == "__main__":
    test_shorten_yaml()
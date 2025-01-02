"""
Script to combine individual product JSON files into a standardized format.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

def load_json_file(file_path: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def standardize_product_data(data: Dict) -> Dict:
    """
    Standardize the product data format.
    Ensures all products have consistent fields.
    """
    standard_fields = {
        'name': data.get('name', ''),
        'generic_name': data.get('generic_name', ''),
        'brand_name': data.get('brand_name', ''),
        'description': data.get('description', ''),
        'indications': data.get('indications', []),
        'dosage': data.get('dosage', {}),
        'side_effects': data.get('side_effects', []),
        'contraindications': data.get('contraindications', []),
        'drug_interactions': data.get('drug_interactions', []),
        'warnings': data.get('warnings', []),
        'storage': data.get('storage', ''),
        'manufacturer': data.get('manufacturer', 'HPL'),
    }
    return standard_fields

def combine_json_files(input_dir: str, output_file: str) -> None:
    """
    Combine all JSON files in the input directory into a single standardized JSON file.
    
    Args:
        input_dir: Directory containing individual product JSON files
        output_file: Path to save the combined JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize combined data structure
    combined_data = {
        'products': [],
        'metadata': {
            'total_products': 0,
            'last_updated': '2024-12-21'
        }
    }
    
    # Process each JSON file in the input directory
    input_path = Path(input_dir)
    for file_path in input_path.glob('*.json'):
        try:
            # Load and standardize product data
            product_data = load_json_file(str(file_path))
            standardized_data = standardize_product_data(product_data)
            
            # Add to combined data
            combined_data['products'].append(standardized_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Update metadata
    combined_data['metadata']['total_products'] = len(combined_data['products'])
    
    # Save combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully combined {len(combined_data['products'])} products into {output_file}")

if __name__ == "__main__":
    # Set paths relative to project root
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "raw"
    output_file = project_root / "data" / "processed" / "products.json"
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_file.parent, exist_ok=True)
    
    # Combine JSON files
    combine_json_files(str(input_dir), str(output_file))

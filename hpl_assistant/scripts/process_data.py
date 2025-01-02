"""
Script to process raw product data and create a standardized JSON file in the processed directory.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

def clean_text(text: str) -> str:
    """Clean text by removing HTML tags and special characters."""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:()\-&]', '', text)
    return text.strip()

def load_and_standardize_data(input_file: str) -> Dict:
    """
    Load the raw product data and standardize its format.
    """
    standardized_data = {
        'medications': [],
        'metadata': {
            'total_products': 0,
            'last_updated': '2024-12-21',
            'source': 'HPL Pharmaceuticals'
        }
    }
    
    try:
        # Load the raw data
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Process each product
        for product_name, product_info in raw_data.items():
            if not isinstance(product_info, dict):
                continue
                
            prescribing_info = product_info.get('prescribing_info', {})
            sections = prescribing_info.get('sections', {})
            
            standardized_product = {
                'name': clean_text(product_info.get('product_name', '')),
                'product_url': product_info.get('product_url', ''),
                'prescribing_info_url': prescribing_info.get('pdf_url', ''),
                'description': clean_text(sections.get('description', '')),
                'composition': clean_text(sections.get('composition', '')),
                'indications': clean_text(sections.get('indications', '')),
                'dosage': clean_text(sections.get('dosage', '') + ' ' + sections.get('administration', '')),
                'side_effects': clean_text(sections.get('side_effects', '')),
                'contraindications': clean_text(sections.get('contraindications', '')),
                'drug_interactions': clean_text(sections.get('drug_interaction', '')),
                'warnings': clean_text(sections.get('warnings_and_precautions', '')),
                'storage': clean_text(sections.get('storage', '')),
                'manufacturer': 'HPL',
                'presentation': clean_text(sections.get('presentation', '')),
                'therapeutic_class': clean_text(sections.get('therapeutic_class', '')),
            }
            
            # Only add products with actual content
            if any(value for value in standardized_product.values() if isinstance(value, str) and value.strip()):
                standardized_data['medications'].append(standardized_product)
                print(f"Processed: {standardized_product['name']}")
    
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise
    
    # Update metadata
    standardized_data['metadata']['total_products'] = len(standardized_data['medications'])
    
    return standardized_data

def save_processed_data(data: Dict, output_file: str) -> None:
    """
    Save the standardized data to the processed directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the standardized data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully processed {data['metadata']['total_products']} medications")
    print(f"Saved to: {output_file}")

def main():
    # Set paths relative to project root
    project_root = Path(__file__).parent.parent
    input_file = project_root / "datasets" / "raw" / "products.json"
    output_file = project_root / "datasets" / "processed" / "medications.json"
    
    try:
        # Load and standardize data
        print(f"Processing data from: {input_file}")
        standardized_data = load_and_standardize_data(str(input_file))
        
        # Save processed data
        save_processed_data(standardized_data, str(output_file))
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main()

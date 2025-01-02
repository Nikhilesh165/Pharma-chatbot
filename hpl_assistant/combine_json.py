import os
import json
from pathlib import Path

def combine_json_files(source_dir: str, output_dir: str) -> None:
    """
    Combines all JSON files from source_dir into a single JSON file in output_dir.
    
    Args:
        source_dir: Directory containing individual JSON files
        output_dir: Directory where the combined JSON file will be saved
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize empty list to store all products
    all_products = []
    
    # Read all JSON files from source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(source_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    product_data = json.load(f)
                    # If product_data is a dictionary, wrap it in a list
                    if isinstance(product_data, dict):
                        product_data = [product_data]
                    all_products.extend(product_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error reading {filename}: {e}")
                continue
    
    # Save combined data to output file
    output_file = os.path.join(output_dir, 'all_products.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        print(f"Successfully combined {len(all_products)} products into {output_file}")
    except Exception as e:
        print(f"Error writing combined file: {e}")

def main():
    # Define source and output directories
    source_dir = os.path.join(os.path.dirname(__file__), 'raw_data')
    output_dir = os.path.join(os.path.dirname(__file__), 'processed')
    
    # Combine JSON files
    combine_json_files(source_dir, output_dir)

if __name__ == "__main__":
    main()

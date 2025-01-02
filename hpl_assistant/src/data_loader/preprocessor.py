"""
Data preprocessing module for cleaning and structuring HPL pharmaceutical data.
"""

import json
import os
import re
import logging
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HPLDataPreprocessor:
    """Preprocesses HPL pharmaceutical data for the RAG system."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing raw JSON files
            output_dir: Directory to save processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML artifacts
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up common artifacts
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        # Remove citation markers
        text = re.sub(r'\(\d+\)', '', text)
        
        return text.strip()
    
    def extract_sections(self, prescribing_info: Dict[str, Any]) -> Dict[str, str]:
        """Extract and clean prescribing information sections."""
        cleaned_info = {}
        
        if 'sections' in prescribing_info:
            for section, content in prescribing_info['sections'].items():
                # Clean the section content
                cleaned_content = self.clean_text(content)
                
                # Skip empty sections or "Not Found" sections
                if cleaned_content and cleaned_content.lower() != "not found":
                    # Convert section name to a standardized format
                    section_name = section.lower()
                    cleaned_info[section_name] = cleaned_content
                
        return cleaned_info
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize the filename to be filesystem-friendly."""
        # Remove special characters and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        # Convert to lowercase
        return name.lower()

    def process_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single product entry."""
        try:
            name = product_data.get('product_name', '')
            if not name:
                return None
            
            processed_data = {
                'name': name,
                'url': product_data.get('product_url', ''),
                'prescribing_info': {}
            }
            
            # Process prescribing information if available
            if 'prescribing_info' in product_data:
                prescribing_info = product_data['prescribing_info']
                processed_data['prescribing_info'] = {
                    'pdf_url': prescribing_info.get('pdf_url', ''),
                    'sections': self.extract_sections(prescribing_info)
                }
            
            # Save individual product file
            output_path = self.output_dir / f"{self.sanitize_filename(name)}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing product {name}: {str(e)}")
            return None

    def process_all_products(self) -> None:
        """Process all products from the input JSON file."""
        try:
            # Read the main products file
            with open(self.input_dir / "products.json", 'r', encoding='utf-8') as f:
                products_data = json.load(f)
            
            processed_count = 0
            error_count = 0
            
            # Process each product
            for product_name, product_data in products_data.items():
                logging.info(f"Processing {product_name}")
                
                if self.process_product(product_data):
                    processed_count += 1
                else:
                    error_count += 1
            
            logging.info(f"Processing complete. Processed {processed_count} products. Errors: {error_count}")
            
        except Exception as e:
            logging.error(f"Error reading products file: {str(e)}")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics about the processed data."""
        stats = {
            'total_files': 0,
            'sections_present': {},
            'avg_section_length': {},
            'empty_sections': {}
        }
        
        # Analyze all processed files
        for file_path in self.output_dir.glob('*.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats['total_files'] += 1
            
            # Analyze prescribing information sections
            if 'prescribing_info' in data and 'sections' in data['prescribing_info']:
                for section, content in data['prescribing_info']['sections'].items():
                    # Track section presence
                    stats['sections_present'][section] = stats['sections_present'].get(section, 0) + 1
                    
                    # Track section lengths
                    if content:
                        current_len = stats['avg_section_length'].get(section, [0, 0])
                        stats['avg_section_length'][section] = [
                            current_len[0] + len(content),
                            current_len[1] + 1
                        ]
                    else:
                        stats['empty_sections'][section] = stats['empty_sections'].get(section, 0) + 1
        
        # Calculate averages
        for section, (total_len, count) in stats['avg_section_length'].items():
            stats['avg_section_length'][section] = round(total_len / count, 2)
        
        return stats

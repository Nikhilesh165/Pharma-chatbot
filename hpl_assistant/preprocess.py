from src.data_loader.preprocessor import HPLDataPreprocessor
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
preprocessor = HPLDataPreprocessor(
    os.path.join(base_dir, 'datasets', 'raw'),
    os.path.join(base_dir, 'datasets', 'processed')
)
preprocessor.process_all_products()

# HPL Pharmaceutical Knowledge Assistant

A RAG-based pharmaceutical knowledge assistant specifically designed for HPL pharmaceutical products. This assistant helps users understand medication details, find information about drug interactions, dosages, and more.

## Features

- Question Answering about HPL pharmaceutical products
- Detailed prescribing information retrieval
- Section-specific information search (dosage, side effects, etc.)
- PDF-based information extraction
- Command-line interface for easy interaction

## Project Structure

```
.
├── datasets/               # JSON datasets
│   ├── raw/               # Raw scraped data
│   └── processed/         # Cleaned and structured data
├── src/
│   ├── agents/            # RAG agent implementation
│   ├── data_loader/       # Dataset preprocessing
│   ├── embeddings/        # Vector store implementation
│   └── cli.py            # Command-line interface
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Process the raw data:
```python
from src.data_loader.preprocessor import HPLDataPreprocessor

preprocessor = HPLDataPreprocessor(
    input_dir="datasets/raw",
    output_dir="datasets/processed"
)
preprocessor.process_all_products()
```

3. Start the CLI interface:
```bash
python src/cli.py
```

## Usage

The assistant can answer questions about HPL pharmaceutical products such as:
- "What are the side effects of [medication]?"
- "What is the recommended dosage for [condition]?"
- "List the contraindications for [medication]"
- "What are the drug interactions for [medication]?"

## Components

1. Data Processing
   - Handles PDF extraction and text processing
   - Structures information into standardized sections
   - Cleans and normalizes text content

2. Vector Store
   - Uses FAISS for efficient similarity search
   - Sentence transformer embeddings
   - Chunked text storage for better retrieval

3. RAG System
   - LangChain for document processing
   - Ollama integration for language model
   - Context-aware response generation

## Notes

- This system is specifically designed for HPL pharmaceutical products
- Information is sourced from official prescribing information PDFs
- The system maintains source attribution for all information
- Responses are generated based on retrieved context only

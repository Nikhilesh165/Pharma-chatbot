# Pharmaceutical Knowledge Assistant

A RAG-based pharmaceutical knowledge assistant specifically designed for pharmaceutical products. This assistant helps users understand medication details, find information about drug interactions, dosages, and more.

## Features

- Question Answering about pharmaceutical products
- Detailed prescribing information retrieval
- Section-specific information search (dosage, side effects, etc.)
- PDF-based information extraction
- Command-line interface for easy interaction
- **Advanced Retrieval-Augmented Generation (RAG)**:
  - Enhanced query analysis to understand user intent and extract key information.
  - Multi-query retrieval for generating alternative queries to improve information retrieval.
  - Document reranking based on relevance to user queries using advanced scoring techniques.
  - Structured answer synthesis that provides clear, well-reasoned responses with confidence scores and medical disclaimers.
- **Streamlit Interface**:
  - A user-friendly web interface for interacting with the chatbot, allowing users to ask questions and receive answers in real-time.
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

3. Start the Streamlit interface:
```bash
streamlit run app.py
```
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
   - Implements advanced query analysis for understanding user intent.
   - Supports multi-query retrieval to generate alternative queries.
   - Reranks documents based on relevance using sophisticated scoring techniques.
   - Synthesizes structured answers with confidence scores and medical disclaimers.

4. Streamlit Interface
   - Provides a user-friendly web interface for real-time interaction with the chatbot.

## Notes

- This system is specifically designed for pharmaceutical products
- Information is sourced from official prescribing information PDFs
- The system maintains source attribution for all information
- Responses are generated based on retrieved context only

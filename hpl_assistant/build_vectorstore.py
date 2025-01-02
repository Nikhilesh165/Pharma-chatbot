from src.embeddings.vectorstore import HPLVectorStore
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
vector_store = HPLVectorStore(
    processed_data_dir=os.path.join(base_dir, 'datasets', 'processed'),
    embeddings_dir=os.path.join(base_dir, 'embeddings')
)

# Build and save the vector store
vector_store.build()

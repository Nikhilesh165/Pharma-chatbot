"""
Test script to run a single query through the HPL Assistant.
"""

import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.advanced_rag_agent import AdvancedHPLRagAgent

def main():
    try:
        # Initialize agent
        print("Initializing HPL Pharmaceutical Knowledge Assistant...")
        agent = AdvancedHPLRagAgent(model_name="llama2")
        
        # Test query
        question = "What are the side effects of Acecard?"
        print(f"\nQuestion: {question}\n")
        
        # Get answer
        result = agent.query(question)
        
        # Print results
        print("\nAnswer:", result["answer"])
        print("\nConfidence:", f"{result['confidence']:.2%}")
        
        if result["reasoning"]:
            print("\nReasoning Path:")
            for step in result["reasoning"]:
                print(f"- {step}")
        
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
        
        if result["disclaimer"]:
            print(f"\nMedical Disclaimer: {result['disclaimer']}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

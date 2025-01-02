"""
Command-line interface for the HPL Pharmaceutical Knowledge Assistant.
"""

import os
import sys
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.advanced_rag_agent import AdvancedHPLRagAgent

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='HPL Pharmaceutical Knowledge Assistant - Ask questions about HPL pharmaceutical products'
    )
    parser.add_argument(
        '--model', 
        type=str,
        default='llama2',
        help='Name of the Ollama model to use (default: llama2)'
    )
    args = parser.parse_args()

    try:
        # Initialize agent
        print("Initializing HPL Pharmaceutical Knowledge Assistant...")
        agent = AdvancedHPLRagAgent(model_name=args.model)
        
        print("\nWelcome to the HPL Pharmaceutical Knowledge Assistant!")
        print("Ask questions about HPL pharmaceutical products, or type 'quit' to exit.")
        print("\nExample questions:")
        print("- What are the side effects of [medication]?")
        print("- What is the recommended dosage for [condition]?")
        print("- List the contraindications for [medication]")
        print("- What are the drug interactions for [medication]?")
        
        while True:
            try:
                # Get user input
                question = input("\nEnter your question (or 'quit' to exit): ")
                question = question.strip()
                
                # Check for quit command
                if question.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                
                # Skip empty input
                if not question:
                    continue
                
                # Process the question
                result = agent.query(question)
                
                # Print the results
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
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nInput stream closed. Exiting...")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Please try rephrasing your question.")
                continue
    
    except KeyboardInterrupt:
        print("\nExiting during initialization...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

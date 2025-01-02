from src.agents.rag_agent import HPLRagAgent

def main():
    # Initialize agent
    print("Initializing HPL Pharmaceutical Knowledge Assistant...")
    agent = HPLRagAgent(model_name="llama2")
    
    # Get answer
    question = "what are the side effects of albicon"
    print("\nQuestion:", question)
    result = agent.query(question)
    
    # Print answer and sources
    print("\nAnswer:")
    print(result['answer'])
    print("\nSources:")
    for source in result['sources']:
        print(f"- {source}")

if __name__ == "__main__":
    main()

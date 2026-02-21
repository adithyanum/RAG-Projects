from engine import ResearchAgent

def main():
    # Instantiate the brain
    agent = ResearchAgent()
    
    print("--- 🤖 Agentic RAG Loaded ---")
    
    while True:
        user_query = input('\nType to ask, Type exit to quit :\t ')
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        response = agent.handle_turn(user_query)
        
        print(f"\n{response}")

if __name__ == "__main__":
    main()
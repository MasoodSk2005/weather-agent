import os
import sys
from dotenv import load_dotenv

from data_loader import WeatherDataLoader
from agent import WeatherAgent

# Load environment variables from .env file
load_dotenv()

def main():
    print("=" * 60)
    print("      MULTI-STEP WEATHER AI AGENT INITIALIZATION        ")
    print("=" * 60)

    # 1. Verify API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.")
        print("Please ensure your .env file is set up correctly.")
        sys.exit(1)

    # 2. Initialize Data Layer
    print("Loading local weather database...")
    try:
        data_loader = WeatherDataLoader()
        cities = data_loader.list_available_cities()
        print(f"[Success] Database loaded. Grounded Cities: {', '.join(cities)}")
    except Exception as e:
        print(f"[Fatal] Data loader initialization failed: {e}")
        sys.exit(1)

    # 3. Initialize Agent
    print("Initializing ReAct Agent execution loop...")
    try:
        agent = WeatherAgent(data_loader=data_loader)
        print("[Success] Agent initialized and ready.")
    except Exception as e:
        print(f"[Fatal] Agent initialization failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print("\nAgent is ready! Type 'exit' or 'quit' to terminate.")
    print("Try queries like:")
    print("- 'What is the average temperature in London?'")
    print("- 'Compare the total rainfall in New York and Tokyo.' (Testing grounding)")
    print("- 'Find the max temperature in Chicago and summarize the weather there.'\n")

    # 4. Interactive CLI Loop
    while True:
        try:
            user_input = input("\nUser Query >> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting agent workflow. Goodbye!")
                break

            print("\nExecuting Multi-Step Plan...")
            # Run the agent
            result = agent.run(user_query=user_input)

            # Print the execution trace
            agent.print_execution_trace()

            # Print Final LLM Answer
            print("FINAL AGENT RESPONSE:")
            print("-" * 30)
            print(result["final_response"])
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting agent workflow. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Runtime Error] {e}")

if __name__ == "__main__":
    main()
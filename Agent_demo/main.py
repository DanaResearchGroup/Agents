from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

from dotenv import load_dotenv
import tools
import chem_tools
import os
import sys

# Load environment variables
load_dotenv()

def get_model():
    provider = os.getenv("AI_PROVIDER", "google").lower()
    
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or "your_gemini_key" in api_key:
            print("Error: GOOGLE_API_KEY is missing or invalid in .env")
            return None
        return GoogleModel(
            "gemini-2.5-pro",
            api_key=api_key
        )
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        
        if not api_key or "your_openai_key" in api_key:
            # Some local LLMs might not need a key, but usually it's required for auth
            # Checks for localhost usually imply local inference which might not strictly need it,
            # but best to warn.
            if not base_url or ("localhost" not in base_url and "127.0.0.1" not in base_url):
                 print("Error: OPENAI_API_KEY is missing or invalid in .env")
                 return None
        
        # Ensure environment variables are set for the underlying library
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
            
        return OpenAIChatModel(model_name)
    
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
        
        if not api_key or "your_key" in api_key:
             print("Error: DEEPSEEK_API_KEY is missing or invalid in .env")
             return None
        
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
            
        return OpenAIChatModel(model_name)

    else:
        print(f"Error: Unknown AI_PROVIDER '{provider}'. Use 'google', 'openai', or 'deepseek'.")
        return None

def main():
    model = get_model()
    if not model:
        sys.exit(1)

    # Define the agent
    agent = Agent(model,
                  system_prompt="You are an experienced computational chemist agent. You can read, list, and rename files. You can also analyze molecules, validate SMILES, and search for substructures using RDKit.",
                  tools=[tools.read_file, tools.list_files, tools.rename_file, chem_tools.validate_smiles, chem_tools.get_molecule_descriptors, chem_tools.find_substructure, chem_tools.generate_2d_image])

    history = []
    print("AI Agent CLI Started. Type 'quit', 'exit', or Ctrl+C to stop.")
    print("Ensure you have a './test' directory and a valid .env file.")
    
    # Create test directory if not exists, just in case
    if not os.path.exists("./test"):
        os.makedirs("./test")
        print("Created './test' directory.")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # Run the agent
            resp = agent.run_sync(user_input, message_history=history)
            
            # Update history with the new messages (user + assistant + tool interactions)
            history = list(resp.all_messages())
            
            # Print the agent's final response
            # Debugging: Checking attributes of resp object
            # print(f"DEBUG: resp attributes: {dir(resp)}")
            
            # Fallback for different pydantic-ai versions
            if hasattr(resp, 'data'):
                print(f"Agent: {resp.data}")
            elif hasattr(resp, 'output'):
                print(f"Agent: {resp.output}")
            else:
                 # Try to print the object itself if unknown
                 print(f"Agent: {resp}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()

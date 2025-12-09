## This tests how well we can reconstruct scenarios from stored raw event logs.
## We will test various LLMs for the same task and compare the outputs.

import os
import sys
from pathlib import Path

# Add project root to sys.path so imports work when running this file directly
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.llm import Config, get_available_models, LLMClient

def test_scenario_reconstruction(model: str, path_to_log: str):
    # Initialize LLM client
    llm_client = LLMClient(model)

    # Resolve relative path from project root
    log_path = Path(path_to_log)
    if not log_path.is_absolute():
        log_path = project_root / log_path

    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    with open(log_path, 'r') as f:
        log_data = f.read()

    if not llm_client or not log_data:
        print("LLM client or log data not available.")
        return

    # For demonstration, we will just print the selected model
    print(f"Testing scenario reconstruction with model: {model}")



    #produce_layer1ֿ

    # Placeholder for actual reconstruction logic
    # reconstructed_scenario = reconstruct_scenario(log_data, model_id, client)
    
    # Placeholder assertion for successful reconstruction
    # assert reconstructed_scenario is not None, "Scenario reconstruction failed."


def main():
    Config.load()  # Ensure config is loaded
    available_models = get_available_models()
    print("Select a model to test scenario reconstruction:\n" +
          "\n".join([f"{i+1}. {name} ({model_id})" for i, (name, model_id) in enumerate(available_models.items())]) +
          "\nEnter the number of the model to test: ")
    choice = int(input().strip()) - 1
    model_name, model_id = list(available_models.items())[choice]

    print("Enter scenario log file: ")
    path_to_log = input().strip()
    test_scenario_reconstruction(model_id, path_to_log)


if __name__ == "__main__":
    main()

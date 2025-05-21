# -*- coding: utf-8 -*-
"""address_agent.py

This script is designed to normalize and parse address information using a pre-trained SentenceTransformer model and a locality database.
"""

import pandas as pd
import config
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Unused imports like BedrockLLM (langchain_aws), Process (crewai), 
# DuckDuckGoSearchRun, DuckDuckGoSearchResults (langchain.tools), and os have been removed.
from crewai import Agent, Task, Crew 
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
import json
from typing import Any, Tuple

class Normalizer:
    """
    Handles the normalization of address localities using sentence embeddings.

    This class loads a locality database, encodes the localities into embeddings,
    and provides a method to find the most similar localities for a given input string.

    Attributes:
        model (SentenceTransformer): The sentence transformer model used for embeddings.
        locality_data (pd.DataFrame): A DataFrame containing locality information.
                                     Expected to have a "Locality" column.
        locality_embeddings (np.ndarray): A numpy array of embeddings for all localities.
    """
    def __init__(self, hyd_locality_db: str):
        """
        Initialize the Normalizer class.

        Args:
            hyd_locality_db (str): Path to the CSV file containing locality data.
                                   The CSV should have a column named "Locality".
        """
        self.model = SentenceTransformer(config.NORMALIZER_MODEL)
        try:
            self.locality_data = pd.read_csv(hyd_locality_db, encoding="utf-8")
            if "Locality" not in self.locality_data.columns:
                raise ValueError("Locality column not found in the provided CSV file.")
            self.locality_embeddings = self.model.encode(self.locality_data["Locality"].astype(str).tolist())
        except FileNotFoundError:
            print(f"Error: Locality database file not found at {hyd_locality_db}")
            # Or raise a custom exception
            self.locality_data = pd.DataFrame(columns=["Locality"]) # Empty dataframe
            self.locality_embeddings = np.array([])
        except Exception as e:
            print(f"Error loading or processing locality data: {e}")
            self.locality_data = pd.DataFrame(columns=["Locality"])
            self.locality_embeddings = np.array([])


    def normalize_locality(self, locality: str) -> list[tuple[str, float]]:
        """
        Normalize a locality name by finding the most similar localities in the dataset.

        Args:
            locality (str): The locality name to normalize.

        Returns:
            list[tuple[str, float]]: A list of tuples, where each tuple contains a
                                     normalized locality name (str) and its similarity
                                     score (float). Returns an empty list if an error
                                     occurs during normalization or if the input locality is empty.
        """
        if not locality: # Handle empty input string
            return []
        
        if self.locality_embeddings.size == 0: # Check if embeddings are available
            print("Warning: Locality embeddings are not available. Cannot normalize.")
            return []

        try:
            address_embedding = self.model.encode([locality])
            similarities = cosine_similarity(address_embedding, self.locality_embeddings)
            
            # Ensure similarities is 2D and get the first row
            similarity_scores_row = similarities[0] if similarities.ndim == 2 else similarities
            
            similarity_scores = list(zip(self.locality_data["Locality"], similarity_scores_row))
            sorted_localities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            return sorted_localities[:config.TOP_N_LOCALITIES]
        except Exception as e:
            print(f"Error normalizing locality: {locality} - {e}")
            return []

class AddressOutputParser(BaseModel):
    """
    Defines the expected structure for the output of the address parsing agent.

    This Pydantic model ensures that the agent's output includes a 'locality' field.
    The description for the field guides the LLM on how to populate it.

    Attributes:
        locality (str): The locality identified by the model. Should be "None" if
                        no locality is confidently identified.
    """
    locality: str = Field(description="Locality found by the model. If no locality is confidently identified, this should be 'None'.")

def setup_agents() -> Agent:
    """
    Initializes and returns the Address Parsing Agent.

    The agent is configured with a specific role, goal, backstory, LLM
    (loaded from config), and is set to not allow delegation.

    Returns:
        Agent: The configured Address Parsing Agent.
    """
    address_parser = Agent(
        role='Address Parsing Agent',
        goal='Accurately identify the correct locality from a given address, a list of locality options, and any additional contextual information provided. If the locality cannot be determined with high confidence, return "None"',
        backstory='''
        You are an Address Parsing Specialist with extensive expertise in analyzing and interpreting address formats across various regions. Your deep understanding of geographical hierarchies, address structures, and contextual clues enables you to pinpoint the correct locality from a additional information and provided list . You prioritize precision and avoid assumptions, ensuring that only highly confident matches are returned. When uncertainty arises, you default to "None" to maintain accuracy and reliability.
        ''',
        verbose=True,
        allow_delegation=False,
        llm=config.ADDRESS_PARSER_LLM,
        tools=[],
    )
    return address_parser

def setup_tasks(address_parser: Agent) -> Task:
    """
    Creates and returns the address parsing task.

    The task defines the description for the agent, the expected output format
    (using AddressOutputParser), the agent responsible for the task, and the
    input variables it requires from the context.

    Args:
        address_parser (Agent): The Address Parsing Agent that will execute this task.

    Returns:
        Task: The configured address parsing task.
    """
    task1 = Task(
        description='''Given the address "{address}", top similar locality {locality} options are available and extra info about the address {extra_details}. Please analyze it and identify the correct locality.''',
        expected_output='''A JSON object with the key "locality" representing the identified locality. If no locality is confidently identified, the value should be "None"''',
        agent=address_parser,
        input_variables=["address", "locality", "extra_details"], # Retained as per current structure
        output_model=AddressOutputParser
    )
    return task1

def parse_address_logic(
    address_string: str,
    normalizer: Normalizer,
    address_parser_agent: Agent,
    parsing_task: Task
) -> Tuple[Any, dict]: # Or more specific type for 'Any' if known
    """
    Orchestrates the address parsing process.

    This involves getting supplementary information via web search, normalizing 
    the locality using the Normalizer, and then using the crewAI agent to 
    parse the address and identify the correct locality.

    Args:
        address_string (str): The input address to parse.
        normalizer (Normalizer): An instance of the Normalizer class.
        address_parser_agent (Agent): The configured Address Parsing Agent.
        parsing_task (Task): The configured address parsing task.

    Returns:
        Tuple[Any, dict]: A tuple containing the output from the crew's execution 
                          (expected to be an instance of AddressOutputParser or its raw equivalent)
                          and a dictionary of usage metrics.
    """
    print(f"Processing address: {address_string}")

    # 1. Get supplementary details from web search
    print("Fetching additional details from DuckDuckGo...")
    search_results = DDGS().text(address_string, max_results=config.DDG_MAX_RESULTS)
    print(f"Search results: {search_results}")

    # 2. Normalize the locality
    print("Normalizing locality...")
    top_localities = normalizer.normalize_locality(address_string)
    if not top_localities:
        print("Warning: Could not normalize locality or no similar localities found.")
    else:
        print(f"Top similar localities: {top_localities}")

    # 3. Setup and run the Crew
    print("Initializing and running the CrewAI agent...")
    crew = Crew(
        agents=[address_parser_agent],
        tasks=[parsing_task],
        verbose=True, # Or configure via config.py e.g. config.CREW_VERBOSE
    )

    # Prepare inputs for the crew
    crew_inputs = {
        "address": address_string,
        "locality": str(top_localities), # Convert list of tuples to string for the prompt
        "extra_details": str(search_results) # Convert list of dicts to string for the prompt
    }

    agent_output = crew.kickoff(inputs=crew_inputs)
    
    print(f"Agent raw output: {agent_output.raw if hasattr(agent_output, 'raw') else agent_output}")

    return agent_output, crew.usage_metrics


def main():
    """
    Main function to run the address parsing script.

    Initializes components, takes user input for an address,
    parses the address, and prints the identified locality and processing costs.
    """
    # Initialization
    print("Initializing normalizer...")
    normalizer = Normalizer(config.LOCALITY_DB_PATH)

    print("Setting up agent...")
    address_parser_agent = setup_agents()
    
    print("Setting up task...")
    parsing_task = setup_tasks(address_parser_agent)

    # User input
    address_data = input("Enter the address: ")

    # Core logic execution
    if address_data:
        # Pass the agent and task instances, not the setup functions
        agent_output, usage_metrics = parse_address_logic(
            address_data,
            normalizer,
            address_parser_agent, # Pass the created agent
            parsing_task         # Pass the created task
        )

        # Standardize Output Processing
        final_locality = "None" # Default value
        if isinstance(agent_output, AddressOutputParser):
            final_locality = agent_output.locality
            print(f"Successfully parsed output. Identified Locality: {final_locality}")
        elif hasattr(agent_output, 'raw') and isinstance(agent_output.raw, dict):
            final_locality = agent_output.raw.get('locality', 'None (from raw dict)')
            print(f"Processed from raw attribute. Identified Locality: {final_locality}")
        elif isinstance(agent_output, dict):
            final_locality = agent_output.get('locality', 'None (from dict)')
            print(f"Processed from dict output. Identified Locality: {final_locality}")
        else:
            # Fallback if the output is not the expected Pydantic model or dict
            print(f"Warning: Output format not as expected. Raw output: {agent_output}")
            # Attempt to deserialize if it's a JSON string representation of AddressOutputParser
            try:
                if isinstance(agent_output, str):
                    data = json.loads(agent_output)
                    if isinstance(data, dict):
                         final_locality = data.get('locality', 'None (from string JSON)')
                elif hasattr(agent_output, 'raw') and isinstance(agent_output.raw, str):
                    data = json.loads(agent_output.raw)
                    if isinstance(data, dict):
                        final_locality = data.get('locality', 'None (from raw string JSON)')
            except json.JSONDecodeError:
                print(f"Could not parse locality from raw output: {agent_output}")
            except Exception as e:
                print(f"An unexpected error occurred during output parsing: {e}")


        print(f"Identified Locality: {final_locality}")

        # Cost calculation
        if usage_metrics:
            costs = (config.COST_PROMPT_TOKEN * usage_metrics.get('prompt_tokens', 0) +
                     config.COST_COMPLETION_TOKEN * usage_metrics.get('completion_tokens', 0))
            print(f"Total estimated costs: ${costs:.4f}")
        else:
            print("Could not retrieve usage metrics.")
    else:
        print("No address entered.")

if __name__ == "__main__":
    main()

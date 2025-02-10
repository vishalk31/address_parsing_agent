# -*- coding: utf-8 -*-
"""address_agent.py

This script is designed to normalize and parse address information using a pre-trained SentenceTransformer model and a locality database.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_aws import BedrockLLM
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
import json

class Normalizer:
    def __init__(self, hyd_locality_db):
        """
        Initialize the Normalizer class with a city-state file.

        Args:
            hyd_locality_db (str): Path to the CSV file containing locality data.
        """
        self.model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
        self.locality_data = pd.read_csv(hyd_locality_db, encoding="utf-8")
        self.locality_embeddings = self.model.encode(self.locality_data["Locality"])

    def normalize_locality(self, locality):
        """
        Normalize a locality name by finding the most similar locality in the dataset.

        Args:
            locality (str): The locality name to normalize.

        Returns:
            list: A list of tuples containing the top N localities and their similarity scores.
        """
        try:
            if locality:
                address_embedding = self.model.encode([locality])
                similarities = cosine_similarity(address_embedding, self.locality_embeddings)
                similarity_scores = list(zip(self.locality_data["Locality"], similarities[0]))
                sorted_localities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
                return sorted_localities[:10]
        except Exception as e:
            print(f"Error normalizing locality: {locality} - {e}")
            return "None"

class AddressOutputParser(BaseModel):
    locality: str = Field(description="Locality found by the model. If no locality is confidently identified, this should be 'None'.")

def setup_agents():
    address_parser = Agent(
        role='Address Parsing Agent',
        goal='Accurately identify the correct locality from a given address, a list of locality options, and any additional contextual information provided. If the locality cannot be determined with high confidence, return "None"',
        backstory='''
        You are an Address Parsing Specialist with extensive expertise in analyzing and interpreting address formats across various regions. Your deep understanding of geographical hierarchies, address structures, and contextual clues enables you to pinpoint the correct locality from a additional information and provided list . You prioritize precision and avoid assumptions, ensuring that only highly confident matches are returned. When uncertainty arises, you default to "None" to maintain accuracy and reliability.
        ''',
        verbose=True,
        allow_delegation=False,
        llm="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        tools=[],
    )
    return address_parser

def setup_tasks(address_parser):
    task1 = Task(
        description='''Given the address "{address}", top similar locality {locality} options are available and extra info about the address {extra_details}. Please analyze it and identify the correct locality.''',
        expected_output='''A JSON object with the key "locality" representing the identified locality. If no locality is confidently identified, the value should be "None"''',
        agent=address_parser,
        input_variables=["address", "locality", "extra_details"],
        output_model=AddressOutputParser
    )
    return task1

def main():
    hyd_locality_db = "hyd_locality.csv"
    normalizer = Normalizer(hyd_locality_db)

    address_data = input("Enter the address: ")
    results = DDGS().text(address_data, max_results=5)
    print(results)

    top_10_localities = normalizer.normalize_locality(address_data)
    print(top_10_localities)

    address_parser = setup_agents()
    task1 = setup_tasks(address_parser)

    crew = Crew(
        agents=[address_parser],
        tasks=[task1],
        verbose=True,
    )

    output = crew.kickoff(inputs={"address": address_data, "locality": str(top_10_localities), "extra_details": str(results)})
    output = json.dumps(output.raw)
    print(output)

    costs = 0.00018 * (crew.usage_metrics.prompt_tokens) / 1000 + 0.00024 * (crew.usage_metrics.completion_tokens) / 1000
    print(f"Total costs: ${costs:.4f}")

if __name__ == "__main__":
    main()

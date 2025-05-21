# config.py

# SentenceTransformer model for address normalization
NORMALIZER_MODEL = "Alibaba-NLP/gte-modernbert-base"

# LLM model for the Address Parsing Agent
ADDRESS_PARSER_LLM = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

# Path to the locality database CSV file
LOCALITY_DB_PATH = "hyd_locality.csv"

# Number of top similar localities to consider
TOP_N_LOCALITIES = 10

# DuckDuckGo search results limit
DDG_MAX_RESULTS = 5

# Cost calculation parameters (example values, adjust as needed)
COST_PROMPT_TOKEN = 0.00018 / 1000  # Cost per prompt token
COST_COMPLETION_TOKEN = 0.00024 / 1000  # Cost per completion token

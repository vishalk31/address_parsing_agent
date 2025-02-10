
# Address Parsing and Normalization Agent

This project is designed to parse and normalize address information using a pre-trained SentenceTransformer model and a locality database. It identifies the correct locality from a given address by comparing it against a list of known localities and using contextual information.

## Features

- **Address Normalization**: Normalizes a given address by finding the most similar locality from a pre-defined database.
- **Address Parsing**: Uses an AI agent to parse and identify the correct locality from the address and additional context.
- **Cost Calculation**: Provides an estimate of the cost incurred for using the AI model.

## Requirements

To run this project, you need the following Python packages:

- `crewai`
- `crewai_tools`
- `langchain_community`
- `langchain_aws`
- `transformers`
- `sentence-transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `duckduckgo-search`
- `pydantic`

You can install all the required packages using the following command:

pip install -r requirements.txt
Setup
Clone the repository:

git clone https://github.com/your-username/address-agent.git
cd address-agent
Install dependencies:


pip install -r requirements.txt
Prepare the locality database:

Ensure you have a CSV file (hyd_locality.csv) containing locality data. The file should have a column named Locality.

Place the file in the same directory as the script or update the path in the code.

Usage
Run the script:

python address_agent.py
Input the address:

When prompted, enter the address you want to normalize and parse.

View the output:

The script will output the top matching localities and the identified locality (if any).

It will also display the estimated cost of running the AI model.

Example

Enter the address: 123 Main St, Hyderabad
Output:

{
  "locality": "Gachibowli"
}
Total costs: $0.0012
File Structure
address_agent.py: The main script for address parsing and normalization.

hyd_locality.csv: A CSV file containing locality data (you need to provide this).

requirements.txt: List of Python dependencies.

Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes.

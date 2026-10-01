import requests
import json
from urllib.parse import urlencode
import pandas as pd
from pathlib import Path

def execute_sparql_query(endpoint, query):
    """Execute SPARQL query via HTTP POST to Fuseki."""
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    data = urlencode({'query': query})

    try:
        response = requests.post(endpoint, data=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:500]}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None


def execute_sparql_update(endpoint, update_query):
    """Execute SPARQL UPDATE query via HTTP POST to Fuseki."""
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = urlencode({'update': update_query})

    try:
        response = requests.post(endpoint, data=data, headers=headers)
        response.raise_for_status()
        print("Update successful.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Update Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:500]}")
        return False


def format_results(json_results):
    """Convert JSON results to pandas DataFrame."""
    if not json_results or 'results' not in json_results:
        return pd.DataFrame()

    bindings = json_results['results']['bindings']
    if not bindings:
        return pd.DataFrame()

    rows = []
    for binding in bindings:
        row = {}
        for var, value in binding.items():
            row[var] = value['value']
        rows.append(row)

    return pd.DataFrame(rows)

queries_dir = "../sparql/"

def load_query(query_file, queries_dir=queries_dir):
    """Load SPARQL query from file."""
    query_path = Path(queries_dir) / query_file
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    with open(query_path, encoding='utf-8') as f:
        return f.read()

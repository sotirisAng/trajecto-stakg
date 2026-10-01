from pathlib import Path

import requests

# Configuration — BIRD indoor museum dataset
fuseki_server = "http://localhost:3030"
graph_name = "bird"
graph_name_inference = f"{graph_name}_inference"
query_endpoint = f"{fuseki_server}/{graph_name}/sparql"
query_inference_endpoint = f"{fuseki_server}/{graph_name_inference}/sparql"
update_endpoint = f"{fuseki_server}/{graph_name}/update"

# Get the project root directory (bird/notebooks/helpers -> project root)
_current_file = Path(__file__).resolve()
BIRD_ROOT = _current_file.parent.parent.parent  # bird/
PROJECT_ROOT = BIRD_ROOT.parent                  # repository root

# Ontology files to upload when creating a new dataset (the BIRD KG was built
# with this subset of the modules)
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
ONTOLOGY_FILES = [
    "imports/sosa.owl",
    "imports/ssn.owl",
    "imports/datacron.owl",
    "modules/poi.ttl",
    "modules/context.ttl",
    "modules/user.ttl",
    "modules/temporal-dwell.ttl",
    "modules/spatial.ttl",
    "modules/patterns.ttl",
    "modules/purpose-skos.ttl",
]
MAIN_ONTOLOGY_FILE = ONTOLOGY_DIR / "trajecto.ttl"


def list_datasets():
    """List all datasets in Fuseki."""
    try:
        response = requests.get(f"{fuseki_server}/$/datasets")
        if response.status_code == 200:
            return [ds['ds.name'].lstrip('/') for ds in response.json()['datasets']]
        return []
    except requests.exceptions.RequestException:
        return []


def dataset_exists(dataset_name):
    """Check if a dataset exists in Fuseki."""
    return dataset_name in list_datasets()


def get_fuseki_db_location():
    """Get the Fuseki database location."""
    import os
    common_paths = [
        "/opt/homebrew/Cellar/fuseki/5.4.0/libexec/run/databases",
        "/opt/homebrew/var/fuseki/databases",
        "/usr/local/fuseki/run/databases",
        os.path.expanduser("~/fuseki/run/databases"),
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return "/opt/homebrew/Cellar/fuseki/5.4.0/libexec/run/databases"


def create_dataset(dataset_name, with_inference=False, base_dataset=None):
    """Create a new dataset in Fuseki."""
    if dataset_exists(dataset_name):
        print(f"Dataset '{dataset_name}' already exists")
        return True

    endpoint = f"{fuseki_server}/$/datasets"
    db_location = get_fuseki_db_location()

    if with_inference and base_dataset:
        base_db_path = f"{db_location}/{base_dataset}"
        config = f"""
PREFIX :       <http://base/#>
PREFIX fuseki: <http://jena.apache.org/fuseki#>
PREFIX ja:     <http://jena.hpl.hp.com/2005/11/Assembler#>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX tdb2:   <http://jena.apache.org/2016/tdb#>

:service_tdb_inference  rdf:type            fuseki:Service;
        rdfs:label                    "TDB2 {dataset_name} with inference";
        fuseki:dataset                :tdb_inference_dataset_readwrite;
        fuseki:name                   "{dataset_name}";
        fuseki:serviceQuery           "query" , "sparql" , "";
        fuseki:serviceReadGraphStore  "get";
        fuseki:serviceReadWriteGraphStore
                "data";
        fuseki:serviceUpdate          "" , "update";
        fuseki:serviceUpload          "upload" .

:tdb_inference_dataset_readwrite rdf:type ja:RDFDataset;
     ja:defaultGraph :inferenceModel
     .

:inferenceModel rdf:type      ja:InfModel;
     ja:reasoner [ ja:reasonerURL <http://jena.hpl.hp.com/2003/OWLMicroFBRuleReasoner> ];
     ja:baseModel :baseModel;
     .

:baseModel rdf:type tdb2:GraphTDB2;
     tdb2:location "{base_db_path}" .
"""
        headers = {'Content-Type': 'text/turtle'}
        response = requests.post(endpoint, data=config, headers=headers)
    else:
        config = f"""
PREFIX :       <http://base/#>
PREFIX fuseki: <http://jena.apache.org/fuseki#>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX tdb2:   <http://jena.apache.org/2016/tdb#>

:service_tdb_all  rdf:type            fuseki:Service;
        rdfs:label                    "TDB2 {dataset_name}";
        fuseki:dataset                :tdb_dataset_readwrite;
        fuseki:name                   "{dataset_name}";
        fuseki:serviceQuery           "query" , "sparql" , "";
        fuseki:serviceReadGraphStore  "get";
        fuseki:serviceReadWriteGraphStore
                "data";
        fuseki:serviceUpdate          "" , "update";
        fuseki:serviceUpload          "upload" .

:tdb_dataset_readwrite
        rdf:type       tdb2:DatasetTDB2;
        tdb2:location  "{db_location}/{dataset_name}" .
"""
        headers = {'Content-Type': 'text/turtle'}
        response = requests.post(endpoint, data=config, headers=headers)

    if response.status_code in [200, 201]:
        print(f"Created dataset '{dataset_name}' (inference={with_inference})")
        return True
    else:
        print(f"Failed to create dataset '{dataset_name}': {response.status_code}")
        print(f"Response: {response.text}")
        return False


def initialize_dataset(dataset_name):
    """Initialize a dataset by making a simple query."""
    try:
        endpoint = f"{fuseki_server}/{dataset_name}/sparql"
        response = requests.post(
            endpoint,
            data={'query': 'ASK { ?s ?p ?o }'},
            headers={'Accept': 'application/json'},
            timeout=10
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def ensure_datasets_exist():
    """Ensure both required datasets exist, creating them if needed."""
    print("Checking Fuseki datasets...")

    try:
        requests.get(f"{fuseki_server}/$/ping", timeout=5)
    except requests.exceptions.RequestException:
        print(f"ERROR: Cannot connect to Fuseki at {fuseki_server}")
        return False

    base_existed = dataset_exists(graph_name)
    success1 = create_dataset(graph_name, with_inference=False)

    if not success1:
        print(f"ERROR: Failed to create base dataset '{graph_name}'")
        return False

    if not base_existed:
        print(f"Initializing TDB2 storage for '{graph_name}'...")
        if initialize_dataset(graph_name):
            print("  TDB2 storage initialized")

        print("\nUploading ontologies to new dataset...")
        if not upload_ontologies(graph_name):
            print("  WARNING: Some ontology uploads failed")

    success2 = create_dataset(
        graph_name_inference,
        with_inference=True,
        base_dataset=graph_name
    )

    if success1 and success2:
        print("\nDatasets ready:")
        print(f"  - {graph_name} (data storage)")
        print(f"  - {graph_name_inference} (with OWL reasoning)")
        return True
    return False


def delete_dataset(dataset_name):
    """Delete a dataset from Fuseki."""
    if not dataset_exists(dataset_name):
        print(f"Dataset '{dataset_name}' does not exist")
        return True

    response = requests.delete(f"{fuseki_server}/$/datasets/{dataset_name}")
    if response.status_code in [200, 204]:
        print(f"Deleted dataset '{dataset_name}'")
        return True
    else:
        print(f"Failed to delete dataset '{dataset_name}': {response.status_code}")
        return False


def upload_file(dataset_name, file_path, content_type=None):
    """Upload a single RDF file to a Fuseki dataset."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"  ERROR: File not found: {file_path}")
        return False

    if content_type is None:
        ext = file_path.suffix.lower()
        content_types = {
            '.ttl': 'text/turtle',
            '.owl': 'application/rdf+xml',
            '.rdf': 'application/rdf+xml',
            '.xml': 'application/rdf+xml',
            '.nt': 'application/n-triples',
            '.n3': 'text/n3',
            '.jsonld': 'application/ld+json',
        }
        content_type = content_types.get(ext, 'application/rdf+xml')

    upload_endpoint = f"{fuseki_server}/{dataset_name}/data"

    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                upload_endpoint,
                data=f,
                headers={'Content-Type': content_type},
                timeout=120
            )

        if response.status_code in [200, 201, 204]:
            return True
        else:
            print(f"  ERROR: Upload failed ({response.status_code}): {response.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Upload request failed: {e}")
        return False


def upload_ontologies(dataset_name=None):
    """Upload all ontology files to a Fuseki dataset."""
    if dataset_name is None:
        dataset_name = graph_name

    if not dataset_exists(dataset_name):
        print(f"ERROR: Dataset '{dataset_name}' does not exist")
        return False

    print(f"Uploading ontologies to dataset '{dataset_name}'...")

    success_count = 0
    fail_count = 0

    for filename in ONTOLOGY_FILES:
        file_path = ONTOLOGY_DIR / filename
        print(f"  Uploading {filename}...", end=" ")

        if upload_file(dataset_name, file_path):
            print("OK")
            success_count += 1
        else:
            fail_count += 1

    if MAIN_ONTOLOGY_FILE.exists():
        print(f"  Uploading {MAIN_ONTOLOGY_FILE.name}...", end=" ")
        if upload_file(dataset_name, MAIN_ONTOLOGY_FILE):
            print("OK")
            success_count += 1
        else:
            fail_count += 1

    print(f"\nUpload complete: {success_count} succeeded, {fail_count} failed")
    return fail_count == 0


def setup_new_dataset():
    """Convenience function to create datasets with ontologies."""
    print("=" * 50)
    print("Setting up BIRD museum dataset")
    print("=" * 50)

    if not ensure_datasets_exist():
        print("\nERROR: Failed to create datasets")
        return False

    print("\n" + "=" * 50)
    print("Setup complete!")
    print(f"  Data endpoint: {fuseki_server}/{graph_name}/sparql")
    print(f"  Inference endpoint: {fuseki_server}/{graph_name_inference}/sparql")
    print("=" * 50)
    return True


print(f"Fuseki query endpoint: {query_endpoint}")
print(f"Fuseki update endpoint: {update_endpoint}")

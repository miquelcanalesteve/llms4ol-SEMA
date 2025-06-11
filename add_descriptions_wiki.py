import json
import requests

def get_wikipedia_summary(concept, lang='en'):
    """
    Fetch the Wikipedia summary for a given concept using the MediaWiki REST API.

    Args:
        concept (str): The concept to search for.
        lang (str): The language edition of Wikipedia (default is 'en').

    Returns:
        str or None: The summary text if found, otherwise None.
    """
    concept_formatted = concept.replace(' ', '_')
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{concept_formatted}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get('extract', '')
    else:
        return None

def load_concepts(file_path):
    """
    Load concepts from a text file, ignoring empty lines.

    Args:
        file_path (str): Path to the input file.

    Returns:
        List[str]: A list of concept names.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

def save_results(results, output_path):
    """
    Save the results dictionary to a JSON file.

    Args:
        results (dict): Dictionary mapping concepts to summaries.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {output_path}")

def main(input_file, output_file):
    concepts = load_concepts(input_file)
    results = {}

    for concept in concepts:
        print(f"🔎 Fetching: {concept}")
        summary = get_wikipedia_summary(concept)
        if summary:
            results[concept] = summary
        else:
            print(f"⚠️  Not found: {concept}")
            results[concept] = None

    save_results(results, output_file)

if __name__ == "__main__":
    main(
        input_file="data/task_c/MatOnto/train_types.txt",
        output_file="data/task_c/MatOnto/concepts_wikipedia.json"
    )

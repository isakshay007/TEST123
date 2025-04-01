import requests
import pandas as pd
import time
from typing import List, Dict


def fetch_stackoverflow_data(pages: int = 1, page_size: int = 100) -> List[Dict]:
    """
    Fetches questions from Stack Overflow using the Stack Exchange API.
    Returns data with columns: ID, Title, Question, Tags.
    """
    all_questions = []
    base_url = "https://api.stackexchange.com/2.3/questions"

    for page in range(1, pages + 1):
        print(f"Fetching page {page}...")  # Debugging statement
        params = {
            'site': 'stackoverflow',
            'pagesize': page_size,
            'page': page,
            'order': 'desc',
            'sort': 'activity',
            'filter': 'withbody',
            'key': ''  # Optional: Add your API key if you have one
        }


        response = requests.get(base_url, params=params)
        print(f"Response status code: {response.status_code}")  # Debugging statement

        if response.status_code == 200:
            data = response.json().get('items', [])
            print(f"Number of questions fetched: {len(data)}")  # Debugging statement
            all_questions.extend(data)
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            break

        time.sleep(1)  # Be nice to the API

    print("Completed fetching all pages.")  # Debugging statement
    return all_questions


def save_data_to_csv(data: List[Dict], file_path: str):
    """
    Save fetched data to a CSV file.
    """
    if not data:
        print("No data to save.")  # Debugging statement
        return

    processed_data = []

    for item in data:
        tag_list = item['tags']

        processed_data.append({
            'ID': item['question_id'],
            'Title': item['title'],
            'Question': item['body'],
            'Tags': ','.join(tag_list)
        })

    df = pd.DataFrame(processed_data)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")  # Debugging statement


if __name__ == "__main__":
    print("Starting data fetching...")  # Debugging statement
    data = fetch_stackoverflow_data(pages=400)

    if data:
        save_data_to_csv(data, 'data/stackoverflow_data.csv')
    
    print("Data fetching completed successfully.")  # Debugging statement

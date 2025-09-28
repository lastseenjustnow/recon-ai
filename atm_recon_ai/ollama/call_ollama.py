import os
from typing import List

import requests

OLLAMA_HOST = "http://127.0.0.1:11434"
API_METHOD = "/api/generate"
MODEL_TYPE = "gemma3:1b"
MODEL_PROMPT = "Explain me this log: "

def check_localhost(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Success! Ollama inference server is up and running at {url}.")
        else:
            print(f"Ollama inference server responded with status code: {response.status_code}")
    except requests.ConnectionError:
        print(f"Failed to connect to {url}. Ollama inference server may be down. Check the requirements in README.md.")


def list_log_files() -> List[str]:
        atm_logs_dir = "./data/atm_logs"
        files = os.listdir(atm_logs_dir)
        return [os.path.join(atm_logs_dir, file_name) for file_name in files]


def analyse_logs():

    log_file_paths = list_log_files()
    if len(log_file_paths) == 0:
        print("Log directory is empty. Insert the logs into the directory and rerun the app.")

    for log_file in list_log_files():
        with open(log_file, 'r', encoding='utf-8') as file:
            content = file.read()

        print("Analysing log file: " + log_file)

        payload = {
            "model": MODEL_TYPE,
            "prompt": MODEL_PROMPT + content,
            "stream": False
        }

        # Send the POST request
        response = requests.post(OLLAMA_HOST + API_METHOD, json=payload)

        # Check the response status code
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()["response"]
            print(response_data)
        else:
            print(f"Error: {response.status_code} - {response.text}")


def run():
    check_localhost(OLLAMA_HOST)
    analyse_logs()

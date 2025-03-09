# helper.py

from together import Together

class ChatBot:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Together(api_key=self.api_key)
        self.history = []

    def append_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def invoke_api(self, model: str = "deepseek-ai/DeepSeek-V3", max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.7, top_k: int = 50, repetition_penalty: float = 1.0, stop: list[str] = ["<｜end▁of▁sentence｜>"]) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=self.history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stream=True
        )
        return self.collapse_response(response)

    def collapse_response(self, response) -> str:
        answer = ""
        for token in response:
            if hasattr(token, "choices"):
                try:
                    answer += token.choices[0].delta.content
                except:
                    pass
        return answer

    def show_history(self) -> None:
        print(self.history)

import re
from datasets import load_dataset, Dataset

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

from datasets import DatasetDict, load_from_disk

def upload_to_hugging_face(dataset_path: str, hf_username: str, repo_name: str, hf_token: str):
    """
    Uploads a dataset to Hugging Face Hub.

    Args:
        dataset_path (str): The local path to the dataset directory.
        hf_username (str): The Hugging Face username.
        repo_name (str): The repository name for the dataset on Hugging Face.
        hf_token (str): The Hugging Face API token for authentication.
    """
    # Load dataset from the saved folder
    dataset_dict = load_from_disk(dataset_path)
    # Convert it into a DatasetDict format
    dataset_dict = DatasetDict({"train": dataset_dict})
    # Authenticate with Hugging Face
    from huggingface_hub import notebook_login
    notebook_login(token=hf_token)
    # Push dataset to Hugging Face
    dataset_dict.push_to_hub(f"{hf_username}/{repo_name}")
    print(f"Dataset pushed to: https://huggingface.co/datasets/{hf_username}/{repo_name}")

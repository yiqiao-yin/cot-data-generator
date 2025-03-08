# run.py

import sys
from helper import ChatBot, extract_xml_answer, extract_hash_answer, upload_to_hugging_face
from datasets import load_dataset, DatasetDict
import time
from tqdm import tqdm

def augment_answer(example: dict, api_key: str) -> dict:
    bot = ChatBot(api_key=api_key)
    bot.history = [{"role": "assistant", "content": "You always provide reasoning. Your answer starts from <think>xxx</think> and <response>."}]
    current_question = example["question"]
    current_answer = example["answer"]
    augmented_content = f"Provide reasoning how to answer question: {current_question} and to arrive with answer: {current_answer}"
    bot.append_history(role="user", content=augmented_content)
    attempts = 0
    while attempts < 5:
        try:
            response = bot.invoke_api()
            example["cot"] = response
            return example
        except Exception as e:
            if attempts < 2:
                time.sleep(10)
            else:
                example["cot"] = "NULL"
                return example
        attempts += 1

def main():
    api_key = input("Enter your Together API Key: ")
    orig_data = input("Enter Huggingface Data ID: ")
    data = load_dataset(orig_data)
    data = data.map(lambda x: augment_answer(x, api_key), num_proc=8)
    new_data_name = input("Enter name for new dataset (used for local dir): ")
    data.save_to_disk(new_data_name)

    # Ask user if they want to upload to Hugging Face
    upload_choice = input("Do you want to upload the dataset to Hugging Face? (yes/no): ")
    if upload_choice.lower() == "yes":
        hf_token = input("Enter your Hugging Face API token: ")
        hf_username = input("Enter your Hugging Face username: ")
        repo_name = input("Enter the desired repository name on Hugging Face: ")
        upload_to_hugging_face("augmented_gsm8k_2k", hf_username, repo_name, hf_token)
    else:
        print("Upload aborted.")

if __name__ == "__main__":
    main()

# run.py

import sys
from helper import ChatBot, upload_to_hugging_face
from datasets import load_dataset
import time
from tqdm import tqdm

def augment_answer(example: dict, api_key: str, question_feature: str, answer_feature: str) -> dict:
    """Augments the answer by using a chatbot to provide reasoning for the answer.

    Args:
        example (dict): The data example to augment.
        api_key (str): API key for the chatbot.
        question_feature (str): The key in the example dict that holds the question text.
        answer_feature (str): The key in the example dict that holds the answer text.

    Returns:
        dict: The augmented example with reasoning added.
    """
    bot = ChatBot(api_key=api_key)
    bot.history = [{"role": "assistant", "content": "You always provide reasoning. Your answer starts from <think>xxx</think> and <response>."}]
    current_question = example[question_feature]
    current_answer = example[answer_feature]
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
    # Gather all user inputs upfront
    api_key = input("Enter your Together API Key: ")
    orig_data = input("Enter Huggingface Data ID: ")
    question_feature = input("What is the name of the feature to be used as 'question'? ")
    answer_feature = input("What is the name of the feature to be used as 'answer'? ")
    new_data_name = input("Enter name for new dataset (used for local dir): ")
    
    # Ask user if they want to upload to Hugging Face
    upload_choice = input("Do you want to upload the dataset to Hugging Face? (yes/no): ")
    hf_token = input("Enter your Hugging Face API token: ")
    hf_username = input("Enter your Hugging Face username: ")
    repo_name = input("Enter the desired repository name on Hugging Face: ")

    # Load dataset and augment data
    data = load_dataset(orig_data)
    data = data.map(lambda x: augment_answer(x, api_key, question_feature, answer_feature), num_proc=8)
    data.save_to_disk(new_data_name)
    if upload_choice.lower() == "yes":
        upload_to_hugging_face(new_data_name, hf_username, repo_name, hf_token)
    else:
        print("Upload aborted.")

if __name__ == "__main__":
    main()

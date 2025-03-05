# Augmenting Data with Together API and Uploading to Hugging Face

This guide provides detailed steps on how to use the Together API to augment a dataset and optionally upload the results to the Hugging Face Hub.

## Overview

The provided Python scripts (`helper.py` and `run.py`) facilitate the augmentation of datasets using the Together LLM model and provide an option to upload the augmented dataset to Hugging Face. This repository is particularly useful for those looking to enhance datasets with additional reasoning or answers generated through AI models.

## Prerequisites

Before you begin, you will need:
- Python 3.8 or higher
- An API key for the Together platform
- (Optional) A Hugging Face account and an API token if you intend to upload the dataset to Hugging Face.

## Installation

1. Clone this repository to your local machine.
2. Install the required Python packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Files

- `helper.py`: Contains helper functions and the `ChatBot` class to interact with the Together API.
- `run.py`: The main script that you run to process the dataset and interact with the command line for optional uploads.

## Running the Script

To run the script, execute the following command in your terminal:

```bash
python run.py
```

You will be prompted to enter:
- Your Together API key
- Whether you want to upload the dataset to Hugging Face
- If yes, your Hugging Face API token, username, and the desired repository name

### Sample CLI Interaction and Output

```plaintext
Enter your Together API Key: [Your API Key Here]
Do you want to upload the dataset to Hugging Face? (yes/no): yes
Enter your Hugging Face API token: [Your Hugging Face Token Here]
Enter your Hugging Face username: [Your Username Here]
Enter the desired repository name on Hugging Face: [Your Repository Name Here]
Dataset pushed to: https://huggingface.co/datasets/[Your Username Here]/[Your Repository Name Here]
```

## Dataset

The script processes and augments the first 2000 data entries from the `openai/gsm8k` dataset available on Hugging Face. After processing, the dataset can be found at the following URL, showcasing a sample where the dataset has already been augmented and uploaded:

[openai-gsm8k-augmented-using-together-ai-deepseek-v3-train-enhanced-2k](https://huggingface.co/datasets/eagle0504/openai-gsm8k-augmented-using-together-ai-deepseek-v3-train-enhanced-2k)

## Conclusion

This setup demonstrates an efficient way to leverage AI to augment datasets and share them on a platform like Hugging Face, enhancing accessibility and collaboration in AI research and development. Follow the prompts in `run.py` to guide you through the process seamlessly.
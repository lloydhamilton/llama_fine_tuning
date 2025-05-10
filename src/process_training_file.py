import json
import os

import pandas as pd
from loguru import logger as log

CURRENT_DIR = os.path.dirname(__file__)


def format_prompt(pre_table: str, table: str, post_table: str, question: str) -> str:
    """Format the prompt for the model inputs."""
    return f"""
Context: {pre_table} \n\n {table} \n\n {post_table}
Question: {question}
"""


def process_file(data_path: str) -> pd.DataFrame:
    """Process the training file and split it into train, validation, and test sets."""
    file_path = os.path.join(CURRENT_DIR, data_path)
    with open(file_path) as file:
        json_data = json.load(file)
    loaded_df = pd.DataFrame(json_data)

    # Extract the "qa" and "annotation" columns
    qa_data = pd.json_normalize(loaded_df["qa"])
    annotation_data = pd.json_normalize(loaded_df["annotation"])
    qa_data = pd.concat([loaded_df, qa_data, annotation_data], axis=1)

    # Select the relevant columns we want to keep
    refined_df = qa_data[
        [
            "table",
            "question",
            "answer",
            "program_re",
            "amt_table",
            "pre_text",
            "post_text",
        ]
    ]

    # Drop rows with NaN values, just to make this easier
    refined_df = refined_df.dropna()

    # Process the input columns with the format_prompt function
    refined_df["inputs"] = refined_df.apply(
        lambda row: format_prompt(
            row["pre_text"], row["table"], row["post_text"], row["question"]
        ),
        axis=1,
    )

    return refined_df


def process_train_file() -> None:
    """Process the training file and split it into train and validation sets."""
    train_df = process_file(os.path.join(CURRENT_DIR, "./data/train.json"))
    sampled_train_df = train_df.sample(frac=0.9, random_state=42)
    sampled_val_df = train_df.drop(sampled_train_df.index)

    # Save the dataframes to CSV files
    sampled_train_df.to_csv(os.path.join(CURRENT_DIR, "./data/train.csv"), index=False)
    sampled_val_df.to_csv(os.path.join(CURRENT_DIR, "./data/val.csv"), index=False)

    log.info("Data processing complete. Train and validation sets saved.")


def process_test_file() -> None:
    """Process the test file."""
    test_df = process_file(os.path.join(CURRENT_DIR, "./data/dev.json"))
    test_df.to_csv(os.path.join(CURRENT_DIR, "./data/test.csv"), index=False)
    log.info("Test data processing complete. Test set saved.")


if __name__ == "__main__":
    process_train_file()
    process_test_file()

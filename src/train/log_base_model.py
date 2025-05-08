import os

import mlflow
from dotenv import load_dotenv
from loguru import logger as log
from mlflow.models import infer_signature
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

current_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(current_dir, "../../.env"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log_base_model(model_path: str) -> None:
    """Log the base model to MLflow."""
    mlflow.set_experiment(model_path)
    with mlflow.start_run():
        # Load the model and tokenizer without padding
        tokenizer_no_pad = AutoTokenizer.from_pretrained(model_path, add_bos_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
        )
        base_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer_no_pad,
            device_map="auto",
        )
        log.info(f"Logging model: {model_path}")
        logged_model = mlflow.transformers.log_model(
            transformers_model=base_pipe,
            task="text-generation",
            artifact_path="model",
            signature=infer_signature(
                model_input={
                    "content": "What is in front of the Notre Dame Main Building?",
                },
            ),
        )
        log.info(
            f"Model logged successfully, model path: {logged_model.model_uri}",
        )


if __name__ == "__main__":
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    log_base_model(model_path)

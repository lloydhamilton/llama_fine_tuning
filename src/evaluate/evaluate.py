import asyncio
import os
import uuid

import mlflow
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers.pipelines.base import Dataset
from mlflow.metrics import rougeLsum

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


class EvaluateLLMModel:

    def __init__(self, model_uri: str, model_name: str):
        """Initialize the EvaluateLLMModel class.

        Args:
            model_uri (str): The URI of the model to evaluate.
        """
        self._model_name = model_name
        self._model_uri = model_uri
        self.val_squad_dataset = load_dataset("squad", split="validation")

    @property
    def model_uri(self) -> str:
        """Return the model URI."""
        return self._model_uri

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def process_dataset(self, data: Dataset) -> pd.DataFrame:
        squad_pd = pd.DataFrame(data[:5])
        answers = pd.json_normalize(squad_pd["answers"])
        squad_pd["answers"] = answers["text"]
        squad_pd["answers"] = squad_pd["answers"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        squad_pd["role"] = "user"
        squad_pd["content"] = squad_pd["question"]
        return squad_pd

    async def eval_experiment(self):
        uid = uuid.uuid4().hex[:6]
        with mlflow.start_run(
            run_name=f"{self.model_name}-qa-{uid}", tags={"id": uid}
        ) as run:
            mlflow.log_param("model_uri", self.model_uri)
            mlflow.log_param("model_name", self.model_name)
            eval_dataset = mlflow.data.from_pandas(
                self.process_dataset(self.val_squad_dataset),
                source="squad_dataset",
                name="squad_dataset",
                targets="answers",
            )
            mlflow.log_input(eval_dataset, context="llm_evaluation")
            mlflow.evaluate(
                model=self.model_uri,
                data=eval_dataset,
                model_type="question-answering",
                evaluators="default",
                extra_metrics=[rougeLsum()],
                # how to load transformer model from mlflow.
                # see -> https://www.mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft/
            )


if __name__ == "__main__":
    # Example usage
    model_uri = "runs:/a283232fc9284556b2ae75053b0318fb/model"  # 1B model
    model_name = "Llama-3.2-1B-Instruct"
    evaluator = EvaluateLLMModel(model_uri, model_name)
    asyncio.run(evaluator.eval_experiment())

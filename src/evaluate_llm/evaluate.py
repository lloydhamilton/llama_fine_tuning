import asyncio
import os
import uuid

import mlflow
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import Pipeline
from transformers.pipelines.base import Dataset

from evaluate_llm.metrics.answer_similarity import answer_similarity

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


class EvaluateLLMModel:

    def __init__(self, model_uri: str, model_name: str):
        """Initialize the EvaluateLLMModel class.

        Args:
            model_uri (str): The URI of the model to evaluate.
        """
        self._model = None
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

    @property
    def model(self) -> Pipeline:
        """Load the model from the specified URI."""
        if self._model is None:
            model = mlflow.transformers.load_model(self._model_uri)
            self._model = model
        return self._model

    def apply_template(self, content: str) -> str:
        """Apply the prompt template to the input content."""
        messages = [{"role": "user", "content": content}]
        return self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def process_dataset(self, data: Dataset) -> pd.DataFrame:
        """Process the dataset for evaluation.

        Also preprocess the dataset with the correct prompt template.
        """
        squad_pd = pd.DataFrame(data[:10])
        answers = pd.json_normalize(squad_pd["answers"])
        squad_pd["answers"] = answers["text"]
        squad_pd["answers"] = squad_pd["answers"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        squad_pd["role"] = "user"
        squad_pd["content"] = squad_pd["question"].apply(self.apply_template)
        return squad_pd

    async def eval_experiment(self) -> None:
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
            mlflow_model = mlflow.transformers.load_model(
                self.model_uri,
            )
            # Configure model params
            mlflow_model.model.generation_config.max_new_tokens = 512
            mlflow_model.model.generation_config.temperature = 0
            mlflow_model.model.generation_config.do_sample = False
            mlflow.evaluate(
                model=mlflow_model,
                data=eval_dataset,
                predictions="outputs",
                extra_metrics=[answer_similarity],
                evaluator_config={"col_mapping": {"inputs": "answers"}},
            )


if __name__ == "__main__":
    # Example usage
    model_uri = "runs:/57be7000b77448e1891c09bb732a8946/model"  # 1B model
    model_name = "Llama-3.2-1B-Instruct"
    evaluator = EvaluateLLMModel(model_uri, model_name)
    asyncio.run(evaluator.eval_experiment())

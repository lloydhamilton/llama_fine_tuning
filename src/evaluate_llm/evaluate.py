import asyncio
import os
import uuid

import mlflow
import pandas as pd
from dotenv import load_dotenv

from evaluate_llm.metrics.answer_similarity import answer_similarity
from train.system_prompt import SYSTEM_PROMPT

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


class EvaluateLLMModel:
    """Class object to encapsulate LLM evaluation."""

    def __init__(self, model_uri: str, model_name: str):
        """Initialize the EvaluateLLMModel class.

        Args:
            model_uri (str): The URI of the model to evaluate.
            model_name (str): The name of the model to evaluate for logging.
        """
        self._model = None
        self._model_name = model_name
        self._model_uri = model_uri
        self.val_squad_dataset = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/val.csv")
        )

    @property
    def model_uri(self) -> str:
        """Return the model URI."""
        return self._model_uri

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @staticmethod
    def apply_template(content: str) -> list[dict[str, str]]:
        """Apply the prompt template to the input content.

        Still need to do this as the model is not able to do this for us.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return messages

    def process_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the dataset for evaluation.

        Also preprocess the dataset with the correct prompt template.
        """
        data["inputs"] = data["inputs"].apply(self.apply_template)
        return data

    async def eval_experiment(self) -> None:
        """Main evaluation entry point."""
        uid = uuid.uuid4().hex[:6]
        mlflow.set_experiment(f"evaluation-{self.model_name}")
        with mlflow.start_run(run_name=f"{self.model_name}-qa-{uid}", tags={"id": uid}):
            mlflow.langchain.autolog()
            mlflow.log_param("model_uri", self.model_uri)
            mlflow.log_param("model_name", self.model_name)
            eval_dataset = mlflow.data.from_pandas(
                self.process_dataset(self.val_squad_dataset),
                source="test.csv",
                name="test.csv",
                targets="program_re",
            )
            mlflow.log_input(eval_dataset, context="llm_evaluation")
            logged_model = mlflow.langchain.log_model(
                lc_model=self.model_uri,
                artifact_path="chain",
                input_example={"inputs": "string"},
            )
            mlflow.evaluate(
                model=logged_model.model_uri,
                data=eval_dataset,
                predictions="outputs",
                extra_metrics=[answer_similarity],
                evaluator_config={"col_mapping": {"inputs": "program_re"}},
            )


if __name__ == "__main__":
    BASE_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "../models/llama-3.2-1b-instruct/langchain_model.py"
    )
    model_name = "Llama-3.2-1B-Instruct"
    evaluator = EvaluateLLMModel(BASE_MODEL_PATH, model_name)
    asyncio.run(evaluator.eval_experiment())

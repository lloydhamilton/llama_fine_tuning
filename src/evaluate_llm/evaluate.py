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

    def __init__(self, model_uri: str, model_name: str, data_type_tag: str | None):
        """Initialize the EvaluateLLMModel class.

        Args:
            model_uri (str): The URI of the model to evaluate.
            model_name (str): The name of the model to evaluate for logging.
            data_type_tag (str): A user generated tag to describe the data type.
        """
        self._model = None
        self._model_name = model_name
        self._model_uri = model_uri
        self._data_type_tag = data_type_tag
        self.val_squad_dataset = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/val.csv")
        )

    @property
    def data_type_tag(self) -> str:
        """Return the data type tag."""
        return self._data_type_tag

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
        mlflow.set_experiment("evaluation-meta-llama/llama-3.2-1b-instruct")
        with mlflow.start_run(
            run_name=f"{self.model_name}-qa-{uid}",
            tags={"id": uid, "model_trained_on": self.data_type_tag},
        ):
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


async def run_evaluation(model_trained_on: str) -> None:
    """Async execution of eval runs."""
    langchain_model_paths = [
        ("../models/llama-3.2-1b-instruct/langchain_model.py", "llama-3.2-1b-instruct"),
        ("../models/lora/langchain_model.py", "llama-3.2-1b-instruct-CovFinQA"),
    ]
    eval_tasks = []
    for path, model_name in langchain_model_paths:
        model_path = os.path.join(os.path.dirname(__file__), path)
        evaluator = EvaluateLLMModel(model_path, model_name, model_trained_on)
        eval_tasks.append(evaluator.eval_experiment())
    await asyncio.gather(*eval_tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLMs")
    parser.add_argument(
        "-t",
        "--model-trained-on-type-tag",
        type=str,
        help="A user generated tag to describe the training data type for the fine"
        "tuned model.",
        required=True,
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.model_trained_on_type_tag))

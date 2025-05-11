import asyncio
import os
import uuid

import mlflow
import pandas as pd
from dotenv import load_dotenv

from evaluate_llm.data_models import ModelRunConfig
from evaluate_llm.metrics.answer_similarity import answer_similarity
from train.system_prompt import SYSTEM_PROMPT

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


class EvaluateLLMModel:
    """Class object to encapsulate LLM evaluation."""

    def __init__(
        self,
        model_uri: str,
        model_name: str,
        input_col: str,
        target_col: str,
    ):
        """Initialize the EvaluateLLMModel class.

        Args:
            model_uri (str): The URI of the model to evaluate.
            model_name (str): The name of the model to evaluate for logging.
            data_type_tag (str): A user generated tag to describe the data type.
            input_col (str): The input column containing context and question.
            target_col (str): The target column containing the answer.
        """
        self._model = None
        self._model_name = model_name
        self._model_uri = model_uri
        self._input_col = input_col
        self._target_col = target_col
        self.val_squad_dataset = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/val.csv")
        )

    @property
    def target_col(self) -> str:
        """Return the target column name."""
        return self._target_col

    @property
    def input_col(self) -> str:
        """Return the target column name."""
        return self._input_col

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
        data["inputs"] = data[self.input_col].apply(self.apply_template)
        return data

    async def eval_experiment(self) -> None:
        """Main evaluation entry point."""
        uid = uuid.uuid4().hex[:6]
        mlflow.set_experiment("evaluation-meta-llama/llama-3.2-1b-instruct")
        params = {
            "id": uid,
            "model_name": self.model_name,
            "input_col": self.input_col,
            "target_col": self.target_col,
        }
        with mlflow.start_run(
            run_name=f"{self.model_name}-{self.input_col}-{self.target_col}-{uid}",
            tags=params,
        ):
            mlflow.langchain.autolog()
            mlflow.log_params(params)
            eval_dataset = mlflow.data.from_pandas(
                self.process_dataset(self.val_squad_dataset),
                source="test.csv",
                name="test.csv",
                targets=self.target_col,
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
                evaluator_config={"col_mapping": {"inputs": self.target_col}},
                model_type="question-answering",
            )


async def run_evaluation(config: ModelRunConfig) -> None:
    """Async execution of eval runs."""
    langchain_model_paths = [
        ("../models/llama-3.2-1b-instruct/langchain_model.py", "base"),
        (config.fine_tune_model_path, "fine-tuned"),
    ]
    eval_tasks = []
    for path, model_name in langchain_model_paths:
        model_path = os.path.join(os.path.dirname(__file__), path)
        evaluator = EvaluateLLMModel(
            model_path, model_name, config.input_col, config.target_col
        )
        eval_tasks.append(evaluator.eval_experiment())
    await asyncio.gather(*eval_tasks)


if __name__ == "__main__":
    asyncio.run(
        run_evaluation(
            ModelRunConfig(
                fine_tune_model_path="../models/train_on_completions_false/"
                "lora-markdown_table_question-program_re-2025-"
                "05-11_19-47/langchain_model.py",
                input_col="markdown_table_question",
                target_col="program_re",
            )
        )
    )
    asyncio.run(
        run_evaluation(
            ModelRunConfig(
                fine_tune_model_path="../models/train_on_completions_false/"
                "lora-pre_table_post_question-program_re-2025"
                "-05-11_20-54/langchain_model.py",
                input_col="pre_table_post_question",
                target_col="program_re",
            )
        )
    )
    asyncio.run(
        run_evaluation(
            ModelRunConfig(
                fine_tune_model_path="../models/train_on_completions_false/"
                "lora-table_question-program_re-2025-"
                "05-11_19-37/langchain_model.py",
                input_col="table_question",
                target_col="program_re",
            )
        )
    )

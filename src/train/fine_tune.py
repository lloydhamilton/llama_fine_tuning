import os
import uuid
from datetime import datetime

import mlflow
import torch
from datasets import (
    Dataset,
    IterableDataset,
    load_dataset,
)
from dotenv import load_dotenv
from loguru import logger as log
from mlflow.models import infer_signature
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from system_prompt import SYSTEM_PROMPT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizer,
    QuantoConfig,
    pipeline,
)
from trl import SFTConfig, SFTTrainer

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))


CURRENT_DIR = os.path.dirname(__file__)
q_cfg = QuantoConfig(weights="int8")


class CustomFineTuner:
    """Custom class for LLM fine-tuning."""

    def __init__(
        self,
        huggingface_model: str,
        lora_config: LoraConfig,
        quant_config: QuantoConfig = None,
        train_on_completions: bool = False,
    ):
        self._huggingface_model_path = huggingface_model
        self._quant_config = quant_config
        self._lora_config = lora_config
        self._train_on_completions = train_on_completions
        self.tokeniser = self.fetch_huggingface_tokenizer()
        self.model = None

    @property
    def huggingface_model(self) -> str:
        """Get the huggingface model."""
        return self._huggingface_model_path

    @property
    def lora_config(self) -> LoraConfig:
        """Get the LoRA config."""
        return self._lora_config

    @property
    def quantization_config(self) -> QuantoConfig:
        """Get the quantization config."""
        return self._quant_config

    @property
    def train_on_completions(self) -> bool:
        """Property to determine if the model is trained on completions."""
        return self._train_on_completions

    @property
    def apply_template(self) -> callable:
        """Get the template to apply to the dataset."""
        if self.train_on_completions:
            return self.apply_prompt_completion_template
        return self.apply_message_template

    @staticmethod
    def fetch_huggingface_data(
        dataset_name: str, split: str = "train"
    ) -> Dataset | IterableDataset:
        """Fetch the dataset from Huggingface."""
        log.info(f"Loading dataset {dataset_name} from Huggingface.")
        hf_dataset = load_dataset(dataset_name, split=split)
        return hf_dataset

    @staticmethod
    def log_trainable_params(model: torch.nn.modules, model_name: str) -> int:
        """Get the number of trainable parameters in the model."""
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Total trainable parameters for {model_name}: {params}")

        return params

    @staticmethod
    def add_lora(model: LlamaForCausalLM) -> torch.nn.modules:
        """Configure the model for LoRA training.

        Target modules can be inspected using: `model.named_modules()`
        """
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        return model

    @staticmethod
    def apply_message_template(
        input_dataset: dict, input_col: str, target_col: str
    ) -> dict:
        """Preprocess the input dataset to fit expected conversational format.

        Training data has the following columns:
        ['table', 'question', 'answer', 'program_re', 'inputs',
        'pre_table_post_question', 'table_question']

        See: https://huggingface.co/docs/trl/en/sft_trainer for details on message
        templates.
        """
        message_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_dataset[input_col]},
            {"role": "assistant", "content": input_dataset[target_col]},
        ]
        return {"messages": message_template}

    @staticmethod
    def apply_prompt_completion_template(
        input_dataset: dict, input_col: str, target_col: str
    ) -> dict:
        """Preprocess the input dataset to fit expected prompt-completion format.

        Training data has the following columns:
        ['table', 'question', 'answer', 'program_re', 'inputs',
        'pre_table_post_question', 'table_question']

        See: https://huggingface.co/docs/trl/en/sft_trainer for details on message
        templates.
        """
        return {
            "prompt": f"{SYSTEM_PROMPT}\n\n{input_dataset[input_col]}",
            "completion": input_dataset[target_col],
        }

    @staticmethod
    def extract_assistant_response(output_text: str) -> str:
        """Extract content after the assistant header."""
        if "<|start_header_id|>assistant<|end_header_id|>" in output_text:
            return output_text.split("<|start_header_id|>assistant<|end_header_id|>")[
                1
            ].strip()
        return output_text

    @staticmethod
    def load_dataset_from_file(file_path: str) -> Dataset:
        """Load dataset from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        dataset = load_dataset("csv", data_files=file_path)
        return dataset

    def load_model_from_checkpoints(self, checkpoint_path: str) -> PeftModel:
        """Load model from checkpoint."""
        model = AutoModelForCausalLM.from_pretrained(
            self.huggingface_model,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            device_map="auto",
        )

        return model

    def fetch_huggingface_tokenizer(self) -> PreTrainedTokenizer:
        """Fetch the tokenizer from Huggingface."""
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.huggingface_model,
            trust_remote_code=True,
        )
        if hf_tokenizer.pad_token_id is None:
            hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id
        return hf_tokenizer

    def fetch_model(self) -> LlamaForCausalLM:
        """Fetch the Llama model from Huggingface.

        If quantization config is provided, use it.
        """
        config = dict(
            pretrained_model_name_or_path=self.huggingface_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            # torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.quantization_config is not None:
            quant_cfg = {
                "quantization_config": self.quantization_config,
            }
            config = config | quant_cfg

        model = AutoModelForCausalLM.from_pretrained(**config)
        if model.config.pad_token_id is None:
            log.info("Setting pad_token_id to eos_token_id")
            model.config.pad_token_id = model.config.eos_token_id
        self.log_trainable_params(model, self.huggingface_model)
        return model

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        input_col: str,
        target_col: str = "program_re",
    ) -> None:
        """Training entry point.

        Args:
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            input_col: The target column name containing the input tokens.
            target_col: The target column name containing the ground truth tokens.

        Steps:
            1. Set MLflow experiment
            2. Fetch model
            3. Preprocess dataset
            4. Finetune the model with LoRA.
        """
        mlflow.set_experiment(f"finetune-{self.huggingface_model}")
        model_to_train = self.fetch_model()
        uid = uuid.uuid4().hex[:6]
        params = {
            "input_col": input_col,
            "target_col": target_col,
            "trained_on_completions": str(self.train_on_completions),
            "model_uri": self.huggingface_model,
        }
        with mlflow.start_run(
            run_name=f"llama-3.2-1b-{input_col}-{target_col}-{uid}",
            tags=params,
        ):
            mlflow.log_params(params=params)
            mlflow.log_params(self.lora_config.to_dict())

            # Process & log training dataset
            processed_train_dataset = train_dataset.map(
                self.apply_template,
                fn_kwargs={
                    "input_col": input_col,
                    "target_col": target_col,
                },
            )
            mlflow_train_dataset = mlflow.data.from_huggingface(
                processed_train_dataset,
                name="train.csv",
                targets=target_col,
            )
            mlflow.log_input(mlflow_train_dataset, context="train_llm_finetuning")

            # Process & log validation dataset
            processed_eval_dataset = val_dataset.map(
                self.apply_template,
                fn_kwargs={
                    "input_col": input_col,
                    "target_col": target_col,
                },
            )
            mlflow_val_dataset = mlflow.data.from_huggingface(
                processed_eval_dataset,
                name="val.csv",
                targets=target_col,
            )
            mlflow.log_input(mlflow_val_dataset, context="val_llm_finetuning")

            # Finetune the model
            log.debug(f"Sample dataset: {processed_train_dataset[0]}")
            model_to_train.train()
            training_args = SFTConfig(
                report_to="mlflow",
                run_name=f"{self.huggingface_model}-finetune-{uuid.uuid4().hex[:6]}",
                output_dir=os.path.join(
                    CURRENT_DIR,
                    "../checkpoints",
                    f"train_on_completions_{str(self.train_on_completions).lower()}",
                    f"lora-{input_col}-{target_col}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
                ),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                per_device_eval_batch_size=1,
                eval_accumulation_steps=1,
                save_steps=10,
                logging_steps=10,
                learning_rate=5e-5,
                max_grad_norm=0.3,
                max_steps=100,
                warmup_ratio=0.03,
                eval_strategy="steps",
                lr_scheduler_type="linear",
                packing=True,
            )
            trainer = SFTTrainer(
                model=model_to_train,
                args=training_args,
                train_dataset=processed_train_dataset,
                eval_dataset=processed_eval_dataset,
                peft_config=self.lora_config,
            )
            trainer.train()

    def mlflow_log_model(self, trainer: SFTTrainer) -> None:
        """Log the model to MLflow."""
        last_run_id = mlflow.last_active_run().info.run_id
        # Save a tokenizer without padding because it is only needed for training
        tokenizer_no_pad = AutoTokenizer.from_pretrained(
            self.huggingface_model, add_bos_token=True
        )
        with mlflow.start_run(run_id=last_run_id):
            mlflow.log_params(self.lora_config.to_dict())
            mlflow.transformers.log_model(
                transformers_model=dict(
                    model=trainer.model, tokenizer=tokenizer_no_pad
                ),
                artifact_path="model",
                signature=infer_signature(
                    model_input={
                        "content": "What is in front of the Notre Dame Main Building?",
                    },
                ),
            )

    def generate_predictions(self, pipeline_model: LlamaForCausalLM, input: str) -> str:
        """Generate predictions from the model using transformer pipelines."""
        message = [
            {"role": "user", "content": input},
        ]
        prompt = self.tokeniser.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        pipe = pipeline(
            "text-generation",
            model=pipeline_model,
            tokenizer=self.tokeniser,
            device_map="auto",
        )
        with torch.no_grad():
            outputs = pipe(prompt, max_new_tokens=512, do_sample=True)
        return self.extract_assistant_response(outputs[0]["generated_text"])


if __name__ == "__main__":

    HUGGING_FACE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    fine_tuner = CustomFineTuner(
        huggingface_model=HUGGING_FACE_MODEL,
        lora_config=lora_cfg,
        train_on_completions=False,
    )
    training_dataset = load_dataset(
        path="csv",
        data_files=os.path.join(CURRENT_DIR, "../data/train.csv"),
        split="train",
    )
    eval_dataset = load_dataset(
        path="csv",
        data_files=os.path.join(CURRENT_DIR, "../data/val.csv"),
        split="train",  # use "train" to load the entire dataset
    )

    # Train the model with the specified dataset and target column,
    # chose of columns are: ['table', 'question', 'answer', 'program_re', 'inputs',
    # 'pre_table_post_question', 'table_question', 'markdown_table_question']
    fine_tune_input_output_params = [
        ("table_question", "program_re"),
        ("markdown_table_question", "program_re"),
        ("pre_table_post_question", "program_re"),
    ]
    for input_val, output_val in fine_tune_input_output_params:
        fine_tuner.train(
            train_dataset=training_dataset,
            val_dataset=eval_dataset,
            input_col=input_val,
            target_col=output_val,
        )

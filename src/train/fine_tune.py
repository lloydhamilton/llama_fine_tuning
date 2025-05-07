import torch
from datasets import (
    Dataset,
    IterableDataset,
    load_dataset,
)
from dotenv import load_dotenv
from loguru import logger as log
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizer,
    QuantoConfig,
)
from trl import SFTConfig, SFTTrainer

load_dotenv("../.env")

# TODO: Updated DVC to be able to download. Write method to get predictions. Early stopping & epochs. MLFLow.

HUGGING_FACE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
q_cfg = QuantoConfig(weights="int8")
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
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


class CustomFineTuner:
    """Custom class for LLM fine-tuning."""

    def __init__(
        self,
        huggingface_model: str,
        lora_config: LoraConfig,
        quant_config: QuantoConfig = None,
    ):
        self._huggingface_model_path = huggingface_model
        self._quant_config = quant_config
        self._lora_config = lora_config
        self.tokenizer = None
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

    @staticmethod
    def fetch_huggingface_data(
        dataset_name: str, split: str = "train"
    ) -> Dataset | IterableDataset:
        """Fetch the dataset from Huggingface."""
        log.info(f"Loading dataset {dataset_name} from Huggingface.")
        hf_dataset = load_dataset(dataset_name, split=split)
        return hf_dataset

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
        """Fetch the Llama model from Huggingface."""
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
            model.config.pad_token_id = model.config.eos_token_id
        self.log_trainable_params(model, self.huggingface_model)
        return model

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
            lora_dropout=0,
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
    def apply_message_template(input_dataset: dict) -> dict:
        """Preprocess the input dataset to fit expected format.

        See: https://huggingface.co/docs/trl/en/sft_trainer for details.
        """
        answer = input_dataset["answers"]["text"]
        if isinstance(input_dataset["answers"]["text"], list):
            answer = input_dataset["answers"]["text"][0]
        message_template = [
            {"role": "system", "content": input_dataset["context"]},
            {"role": "user", "content": input_dataset["question"]},
            {"role": "assistant", "content": answer},
        ]
        return {"messages": message_template}

    def train(self, train_dataset: Dataset) -> None:
        """Training entry point."""
        model_to_train = self.fetch_model()
        processed_dataset = train_dataset.map(self.apply_message_template)
        model_to_train.train()
        training_args = SFTConfig(
            output_dir="cp",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
            # optim = "paged_adamw_32bit",
            save_steps=10,
            logging_steps=10,
            learning_rate=5e-5,
            max_grad_norm=0.3,
            max_steps=60,
            warmup_ratio=0.03,
            # eval_strategy="steps",
            lr_scheduler_type="linear",
            packing=True,
        )
        trainer = SFTTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=processed_dataset,
            peft_config=self.lora_config,
        )
        trainer.train()


if __name__ == "__main__":
    fine_tuner = CustomFineTuner(
        huggingface_model=HUGGING_FACE_MODEL,
        lora_config=lora_cfg,
    )
    dataset = fine_tuner.fetch_huggingface_data("squad")
    fine_tuner.train(dataset)

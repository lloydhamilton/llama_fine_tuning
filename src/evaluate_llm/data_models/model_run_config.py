from pydantic import BaseModel, Field


class ModelRunConfig(BaseModel):
    """Model run configuration."""

    fine_tune_model_path: str = Field(
        ...,
        description="Path to the file containing the langchain wrapper for the "
        "fine-tuned model.",
    )
    input_col: str = Field(
        ...,
        description="The input column containing context and question.",
    )
    target_col: str = Field(
        ...,
        description="The target column containing the answer.",
    )
    base_model_path: str = Field("../models/llama-3.2-1b-instruct/langchain_model.py")

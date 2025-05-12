# CovFinQA

Main dataset: https://github.com/czyssrs/ConvFinQA

This project focuses on fine tuning an LLM to answer questions related to the
financial domain. The goal is to create a model that can understand and respond to
questions about financial documents, such as tables and reports.

The core aim of the fine-tuning process is to enhance the model's ability to comprehend
financial tables and provide an accurate mathematical formula to answer the questions.

For example, for a question:

What was the percentage increase for teleflex incorporated's market performance from 2014-2015?\n

Table: <table class='wikitable'><tr><td>1</td><td>company / index</td><td>2013</td><td>2014</td><td>2015</td><td>2016</td><td>2017</td><td>2018</td></tr><tr><td>2</td><td>teleflex incorporated</td><td>100</td><td>124</td><td>143</td><td>177</td><td>275</td><td>288</td></tr><tr><td>3</td><td>s&p 500 index</td><td>100</td><td>114</td><td>115</td><td>129</td><td>157</td><td>150</td></tr><tr><td>4</td><td>s&p 500 healthcare equipment & supply index</td><td>100</td><td>126</td><td>134</td><td>142</td><td>186</td><td>213</td></tr></table>

Answer: 15.32%

The LLM is fine-tuned to create a mathematical program as below:

```
divide(subtract(143, 124), 124)
```

## Getting Started on this project

### Tech stack:

- Python
- LangChain
- Ollama
- MLflow
- DVC

There are a few pre-requisites to run this repo:

### Python Environment Setup

To set up the python environment, run the following command:

```bash
uv sync
source .venv/bin/activate
```

This will create a virtual environment and install all the required packages.

### Environment Variables
Create a `.env` file in the root directory with the following variables:

```env
HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
TOKENIZERS_PARALLELISM=false
MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

### Dataset

I have used DVC to store the data in a public S3 bucket.
To download the data for this repo, run the following command:

```bash
dvc pull
```

### Exploring Mlflow

To explore the MLflow experiments, you can run the following command:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

This will start the MLflow server on your local machine. 
You can then access the MLflow UI by navigating to link.

### Building Ollama Models

You can build the ollama models to interact with the LLMs using the following steps:
```bash
make build-ollama
```

### Available models in this project

- llama-table_question-program_re
- llama-pre_table_post_question-program_re
- llama-markdown_table_question-program_re

To run any of these models, you can use the following command:

```bash
ollama run <model_name>:latest
```

## Developing

### Deploying a Model on Ollama

1. Install [Ollama](https://ollama.com)
2. Clone [Llama.ccp](https://github.com/ggerganov/llama.cpp) 
3. Download your model from [Hugging Face](https://huggingface.co/llama-3) in `gguf`
format.
4. Convert to gguf format using `llama.cpp`:
```bash
python ../../../llama.cpp/convert_hf_to_gguf.py llama-3.2-1b-instruct --outfile llama-3.2-1b-instruct.gguf
```
5. Convert LoRA adapter to gguf format using `llama.cpp`:
```bash
python convert_lora_to_gguf.py ../tomorro_llm_tech/src/checkpoints/checkpoint-100/ --outfile ../tomorro_llm_tech/src/models/lora/
```

5. Create Modelfile with the following contents, prompt template and LoRA adapter:
```bash
From llama-3.2-1b-instruct.gguf

TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"
PARAMETER num_keep 24
PARAMETER stop <|start_header_id|>
PARAMETER stop <|end_header_id|>
PARAMETER stop <|eot_id|>
ADAPTER checkpoint-100-F16-LoRA.gguf
```

If you are unsure of the template you can find it in the ollama modelfile with:

```bash
ollama show --modelfile llama3

```

6. Run the following command to create the model:
```bash
ollama create <model_name> -f Modelfile
```
7. Run the model:
```bash
ollama run <model_name>:latest
```


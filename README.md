# Tomorrow FinQA

Main dataset: https://github.com/czyssrs/ConvFinQA

## Dataset

To download the data for this repo, run the following command:

```bash
dvc pull
```

## Deploying Model on Ollama

Ref: 
- https://medium.com/intro-to-artificial-intelligence/deploy-fine-tuned-quantizedmodel-to-ollama-88781bd1151e

1. Install [Ollama](https://ollama.com)
2. Clone [Llama.ccp](https://github.com/ggerganov/llama.cpp) 
3. Download your model from [Hugging Face](https://huggingface.co/llama-3) in `gguf`
format.
4. Convert to gguf format using `llama.cpp`:
```bash
python ../../../llama.cpp/convert_hf_to_gguf.py llama-3.2-1b-instruct --outfile llama-3.2-1b-instruct.gguf
```
5. Create Modelfile with the following content and prompt template:
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


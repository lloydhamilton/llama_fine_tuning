# Variables
.PHONY: run-ollama
run-ollama:
	ollama run base_model_llama:latest

.PHONY: build-ollama
build-ollama:
	@echo "Building Ollama llama-3.2-3b-instruct base model..."
	@(cd src/models/llama-3.2-1b-instruct && \
    	  ollama create llama-3.2-1b-instruct -f Modelfile && \
    	  echo "Building Ollama llama-3.2-1b-instruct CovFinQA finetuned model..." && \
    	  cd ../lora && \
    	  ollama create llama-3.2-1b-instruct-CovFinQA -f Modelfile)

.PHONY: ollama-up-base
ollama-base:
	ollama run llama-3.2-1b-instruct:latest

.PHONY: ollama-up-fine-tuned
ollama-base:
	ollama run llama-3.2-1b-instruct-CovFinQA:latest
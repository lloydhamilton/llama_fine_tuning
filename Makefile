.PHONY: build-ollama
build-ollama:
	@echo "Building Ollama llama-3.2-3b-instruct base model..."
	@cd src/models/llama-3.2-1b-instruct && ollama create llama-3.2-1b-instruct -f Modelfile
	@echo "Building Ollama llama-markdown_table_question-program_re model..."
	@cd src/models/train_on_completions_false/lora-markdown_table_question-program_re-2025-05-11_19-47 && ollama create lloydlmhamilton/llama-markdown_table_question-program_re -f Modelfile
	@echo "Building Ollama llama-pre_table_post_question-program_re model..."
	@cd src/models/train_on_completions_false/lora-pre_table_post_question-program_re-2025-05-11_20-54 && ollama create lloydlmhamilton/llama-pre_table_post_question-program_re -f Modelfile
	@echo "Building Ollama llama-table_question-program_re model..."
	@cd src/models/train_on_completions_false/lora-table_question-program_re-2025-05-11_19-37 && ollama create lloydlmhamilton/llama-table_question-program_re -f Modelfile

.PHONY: ollama-up-base
ollama-base:
	ollama run llama-3.2-1b-instruct:latest

.PHONY: pull-ollama
pull-ollama:
	@echo "Pulling Ollama models..."
	@ollama pull lloydlmhamilton/llama-markdown_table_question-program_re
	@ollama pull lloydlmhamilton/llama-pre_table_post_question-program_re
	@ollama pull lloydlmhamilton/llama-table_question-program_re
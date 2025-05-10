# Variables
MODEL_URI := runs:/c48f992d2c614b12b751174d09f0e8a6/model
DOCKER_NAME := llama-3.2-3b-instruct

# Default target
.PHONY: all
all: build-docker

# Build docker image for the MLflow model
.PHONY: build-docker
build-docker:
	@echo "Building Docker image for $(DOCKER_NAME)..."
	mlflow models build-docker --model-uri "$(MODEL_URI)" --name "$(DOCKER_NAME)"

.PHONY: run-ollama
run-ollama:
	ollama run base_model_llama:latest


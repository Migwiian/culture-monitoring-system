.PHONY: setup data train api test kestra clean

# Install dependencies
setup:
	pip install -r requirements.txt
	pre-commit install

# Validate data
data:
	python src/data/make_dataset.py

# Train model with MLflow (for experimentation)
train:
	mlflow run . --experiment-name "voluntas_culture"

# Run FastAPI service (local)
api:
	uvicorn deployment.api.app:app --host 0.0.0.0 --port 9696 --reload

# Test pipeline
test:
	pytest tests/ -v

# Orchestrate the full pipeline with Kestra
kestra:
	@echo "To run the full pipeline with Kestra:"
	@echo "1. Make sure Kestra is running."
	@echo "2. In the Kestra UI, create a new flow using 'src/orchestration/kestra.yml'."
	@echo "3. Run the flow from the Kestra UI."

# Clean artifacts
clean:
	rm -rf data/processed/* models/* mlruns/

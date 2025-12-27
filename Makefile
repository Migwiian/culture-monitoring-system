.PHONY: setup data train api test deploy clean

# Install dependencies
setup:
	pip install -r requirements.txt
	pre-commit install

# Validate data
data:
	python src/data/make_dataset.py

# Train model (with MLflow)
train:
	mlflow run . --experiment-name "voluntas_culture"

# Run FastAPI service (local)
api:
	uvicorn src/services.api:app --host 0.0.0.0 --port 9696 --reload

# Test pipeline
test:
	pytest tests/ -v

# Deploy to Docker
deploy:
	docker-compose -f deployment/docker-compose.yml up -d --build

# Clean artifacts
clean:
	rm -rf data/processed/* models/* mlruns/

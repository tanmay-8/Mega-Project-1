#!/usr/bin/env bash
set -euo pipefail

# Preprocess CSVs
python3 -m scripts.preprocess_data --input data/creditcard.csv --clients 3 --out-root data --seed 42 --test-size 0.2

# Build server image (clients reuse it)
export COMPOSE_PROJECT_NAME=fedavg_demo
docker compose -f docker/docker-compose.yml build server

# Launch server + 3 clients
exec docker compose -f docker/docker-compose.yml up --no-build --abort-on-container-exit

#!/bin/bash
docker-compose up -d
docker exec -it ollama ollama pull llama3.2
docker exec -it ollama ollama serve
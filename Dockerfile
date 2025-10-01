FROM ollama/ollama
LABEL authors="vladislavacatov"

RUN ollama run llama3.2:3b

ENTRYPOINT ["ollama", "serve"]
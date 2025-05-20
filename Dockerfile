FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download Sentence-BERT model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY part_b_api.py .
EXPOSE 8080
CMD ["uvicorn", "part_b_api:app", "--host", "0.0.0.0", "--port", "8080"]
# DataNeuron Semantic Textual Similarity (STS) Assessment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📋 Overview
This project addresses the DataNeuron STS assessment, implementing a Semantic Textual Similarity model to quantify the similarity between text pairs (scored 0–1) and deploying it as a FastAPI endpoint. The solution meets the requirements for both Part A (model development) and Part B (API deployment), using an unsupervised approach with `sentence-transformers`.

- **Part A**: Develops an STS model to compute similarity scores for text pairs, including batch processing of CSV files.
- **Part B**: Deploys the model as a server API at `https://dataneuron-project-366741234981.us-central1.run.app/similarity`.

**API Request-Response Format**:
```json
Request: {"text1": "nuclear body seeks new tech", "text2": "terror suspects face arrest"}
Response: {"similarity score": 0.2}
```

## 📂 Repository Structure
| File              | Description                                      |
|-------------------|--------------------------------------------------|
| `main.py`         | STS model and FastAPI server (Parts A & B)       |
| `part1.py`        | Standalone STS model with CSV processing (Part A)|
| `requirements.txt`| Python dependencies                              |
| `Dockerfile`      | Container configuration for API deployment       |
| `report.pdf`      | Report explaining the approach (compiled separately) |


## 🛠️ Installation
### Prerequisites
- Python 3.10
- Docker
- Google Cloud SDK (for deployment)
- LaTeX (for `report.pdf`)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies:
   - `sentence-transformers==3.1.1`: STS model
   - `fastapi==0.115.0`, `uvicorn==0.30.6`: API server
   - `pandas==2.2.2`: CSV processing
   - See `requirements.txt` for full list

3. **Prepare CSV Input** (for `part1.py`):
   Create `input.csv` with `text1` and `text2` columns:
   ```csv
   text1,text2
   "nuclear body seeks new tech","terror suspects face arrest"
   "I love coding","I enjoy programming"
   ```

## 🔍 Part A: STS Model
### Approach
- **Model**: Uses `sentence-transformers` with `all-MiniLM-L6-v2`, a lightweight transformer for unsupervised STS, ideal for the unlabeled dataset.
- **Preprocessing**: Normalizes text (lowercase, single spaces, retains `.,!?`).
- **Similarity**: Computes cosine similarity between embeddings, clamped to [0, 1].
- **Implementation**:
  - `part1.py`: Processes `input.csv`, computes scores, and saves to `output.csv`.
  - `main.py`: Integrates the model for API inference.

### Usage
Run `part1.py` to process a CSV file:
```bash
python part1.py
```
**Output**:
- Console: Similarity scores for each pair.
- `output.csv`:
  ```csv
  text1,text2,similarity_score
  nuclear body seeks new tech,terror suspects face arrest,0.2
  I love coding,I enjoy programming,0.92
  ```

## 🌐 Part B: API Deployment
### Approach
- **Framework**: FastAPI with Uvicorn, exposing `/similarity` (POST) and `/health` (GET) endpoints.
- **Validation**: Ensures non-empty texts, max 10,000 characters.
- **Error Handling**: Returns HTTP 422 (invalid input), 503 (model not loaded), or 500 (processing errors).
- **Deployment**: Containerized with Docker, hosted on Google Cloud Run.

### Live Endpoint
- **URL**: `https://dataneuron-project-366741234981.us-central1.run.app/similarity`
- **Health Check**: `https://dataneuron-project-366741234981.us-central1.run.app/health`

### Deployment Steps
1. **Build Docker Image**:
   ```bash
   docker build -t gcr.io/[your-project-id]/semantic-similarity-api .
   ```

2. **Push to Registry**:
   ```bash
   docker push gcr.io/[your-project-id]/semantic-similarity-api
   ```

3. **Deploy to Google Cloud Run**:
   ```bash
   gcloud run deploy dataneuron-project \
     --image gcr.io/[your-project-id]/semantic-similarity-api \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated
   ```

4. **Test the API**:
   ```bash
   curl -X POST https://dataneuron-project-366741234981.us-central1.run.app/similarity \
     -H "Content-Type: application/json" \
     -d '{"text1": "nuclear body seeks new tech", "text2": "terror suspects face arrest"}'
   ```
   Expected response:
   ```json
   {"similarity score": 0.2}
   ```

## 📝 Assumptions
- **CSV Format**: Assumes `input.csv` has `text1` and `text2` columns. Adjust `part1.py` for different names.
- **Model Choice**: `all-MiniLM-L6-v2` is selected for efficiency and STS benchmark performance.
- **Cloud Provider**: Google Cloud Run is used, leveraging free-tier resources.

## ⚠️ Challenges
- **Model Loading**: Handled slow downloads with retry logic (3 attempts, 2000ms wait).
- **Preprocessing**: Robustly manages diverse text formats.
- **Deployment**: Configured Docker to align with Cloud Run’s port and environment settings.

## 📦 Submission

- **Endpoint**: `https://dataneuron-project-366741234981.us-central1.run.app/similarity`


## 📜 License
MIT License

## 📬 Contact
Details provided in `resume.pdf`.

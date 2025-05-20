
# This is the docstring for the script, indicating that it implements a production-ready API
# for calculating semantic similarity between two text inputs with robust error handling.

import os
# Imports the 'os' module to access environment variables, used later for configuring the server port.

import re
# Imports the 're' module for regular expressions, used in text preprocessing for cleaning and normalizing text.

import uuid
# Imports the 'uuid' module to generate unique request IDs for tracking requests in logs.

import logging
# Imports the 'logging' module to configure and handle logging for debugging and monitoring.

from typing import Optional
# Imports 'Optional' from 'typing' to indicate that a variable (e.g., the model) can be None or of a specified type.

import numpy as np
# Imports 'numpy' as 'np' for numerical operations, though not directly used in this code (likely included for potential future extensions).

from fastapi import FastAPI, HTTPException, Request, status
# Imports FastAPI components:
# - 'FastAPI' for creating the API application.
# - 'HTTPException' for raising HTTP errors with specific status codes and messages.
# - 'Request' to access request data in middleware.
# - 'status' for HTTP status codes like 200, 500, etc.

from fastapi.responses import JSONResponse
# Imports 'JSONResponse' to return custom JSON responses, used in error handling in middleware.

from pydantic import BaseModel, Field, validator
# Imports Pydantic components for data validation:
# - 'BaseModel' for defining request/response models.
# - 'Field' for setting field constraints (e.g., min/max length).
# - 'validator' for custom validation logic.

from sentence_transformers import SentenceTransformer, util
# Imports components from 'sentence_transformers':
# - 'SentenceTransformer' for loading and using a pre-trained model for text embeddings.
# - 'util' for utility functions like cosine similarity calculation.

import uvicorn
# Imports 'uvicorn', an ASGI server to run the FastAPI application.

from retrying import retry
# Imports the 'retry' decorator from the 'retrying' library to handle retries for model loading with configurable attempts and delays.

import torch
# Imports 'torch', the PyTorch library, used by 'sentence_transformers' for tensor operations during model inference.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
# Configures the logging system:
# - 'level=logging.INFO': Sets the logging level to INFO, capturing info, warning, and error messages.
# - 'format': Defines a custom log format including timestamp, logger name, log level, request ID, and message.
# - 'handlers': Uses 'StreamHandler' to output logs to the console.

logger = logging.getLogger(__name__)
# Creates a logger instance for the current module, allowing logging with module-specific context.

# Configuration constants
MAX_TEXT_LENGTH = 10000
# Defines the maximum length (in characters) for input texts to prevent excessive input sizes.

MODEL_NAME = "all-MiniLM-L6-v2"
# Specifies the pre-trained model name from 'sentence_transformers' to be used for generating text embeddings.

MODEL_LOAD_RETRIES = 3
# Sets the number of retry attempts for loading the model in case of failures.

MODEL_LOAD_WAIT = 2000  # milliseconds
# Defines the wait time (in milliseconds) between retry attempts for model loading.

app = FastAPI(
    title="DataNeuron Semantic Similarity API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)
# Initializes the FastAPI application:
# - 'title': Sets the API's title for documentation.
# - 'version': Specifies the API version.
# - 'docs_url': Sets the endpoint for Swagger UI documentation to '/docs'.
# - 'redoc_url=None': Disables ReDoc documentation.

class TextPairRequest(BaseModel):
    text1: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    text2: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    # Defines a Pydantic model for validating API request payloads:
    # - 'text1' and 'text2': String fields for the two input texts to compare.
    # - 'Field(...)': Indicates the fields are required (using '...').
    # - 'min_length=1': Ensures each text is at least 1 character.
    # - 'max_length=MAX_TEXT_LENGTH': Enforces the maximum text length defined earlier.

    @validator('text1', 'text2')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v
    # Defines a Pydantic validator for 'text1' and 'text2':
    # - Checks that the input is not empty or whitespace-only after stripping.
    # - Raises a 'ValueError' if invalid, which FastAPI converts to a 422 HTTP error.
    # - Returns the validated value if it passes.

class SimilarityResponse(BaseModel):
    similarity_score: float = Field(..., alias="similarity score", ge=0, le=1)
    # Defines a Pydantic model for the API response:
    # - 'similarity_score': A float field for the cosine similarity score.
    # - 'Field(..., alias="similarity score")': Makes the field required and maps the JSON key "similarity score" to the Python attribute 'similarity_score'.
    # - 'ge=0, le=1': Ensures the score is between 0 and 1 (inclusive).

model: Optional[SentenceTransformer] = None
# Declares a global variable 'model' to hold the SentenceTransformer instance, initially set to None.
# - 'Optional' indicates it can be None or a SentenceTransformer object.

class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'system')
        return True
    # Defines a custom logging filter to add a 'request_id' to log records:
    # - 'getattr(record, 'request_id', 'system')': Adds the 'request_id' to the log record, defaulting to 'system' if not set.
    # - 'return True': Allows all log records to pass through the filter.

# Apply filter to all handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestContextFilter())
# Applies the 'RequestContextFilter' to all handlers of the root logger to ensure every log message includes a request ID.

@app.on_event("startup")
@retry(stop_max_attempt_number=MODEL_LOAD_RETRIES, wait_fixed=MODEL_LOAD_WAIT)
def load_model():
    global model
    try:
        logger.info("Loading model", extra={"request_id": "system"})
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully", extra={"request_id": "system"})
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", extra={"request_id": "system"})
        raise RuntimeError(f"Model initialization failed: {str(e)}")
    # Defines a startup event handler for FastAPI to load the model when the application starts:
    # - '@app.on_event("startup")': Registers the function to run on application startup.
    # - '@retry': Retries the function up to 'MODEL_LOAD_RETRIES' times with 'MODEL_LOAD_WAIT' milliseconds between attempts.
    # - 'global model': Allows modification of the global 'model' variable.
    # - Logs the attempt to load the model.
    # - Loads the SentenceTransformer model specified by 'MODEL_NAME'.
    # - Logs success or failure, raising a 'RuntimeError' if loading fails after retries.

def preprocess_text(text: str) -> str:
    """Text normalization pipeline"""
    try:
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'[^\w\s.,!?]', '', text)
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return text
    # Defines a function to preprocess text inputs:
    # - Converts input to a string, strips whitespace, and converts to lowercase.
    # - Replaces multiple spaces with a single space using regex.
    # - Removes all characters except alphanumeric, whitespace, and specific punctuation (.,!?).
    # - Logs any errors during preprocessing and returns the original text if an error occurs.

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    # Defines an HTTP middleware to add a unique request ID to each incoming request:
    # - Generates a unique 'request_id' using UUID.
    # - Stores the 'request_id' in the request's state for access in logging.

    # Set up logging context
    logger = logging.getLogger(__name__)
    old_factory = logging.getLogRecordFactory()
    # Saves the current log record factory to restore it later.

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record
    # Defines a custom log record factory that adds the 'request_id' to every log record.

    logging.setLogRecordFactory(record_factory)
    # Sets the custom log record factory to include the request ID in logs.

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Internal server error"}
        )
    # Processes the request by calling the next middleware or endpoint handler:
    # - Catches any unhandled exceptions, logs them, and returns a 500 error response.

    finally:
        logging.setLogRecordFactory(old_factory)
    # Restores the original log record factory to prevent affecting other requests.

    return response
    # Returns the response from the endpoint or the error response if an exception occurred.

@app.post(
    "/similarity",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK
)
async def calculate_similarity(request: Request, data: TextPairRequest):
    # Defines the main API endpoint to calculate semantic similarity:
    # - '@app.post("/similarity")': Registers a POST endpoint at '/similarity'.
    # - 'response_model=SimilarityResponse': Validates the response using the 'SimilarityResponse' model.
    # - 'status_code=status.HTTP_200_OK': Sets the default success status code to 200.
    # - Takes a 'Request' object and a 'TextPairRequest' payload.

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    # Checks if the model is loaded; raises a 503 error if not.

    try:
        # Preprocessing
        text1 = preprocess_text(data.text1)
        text2 = preprocess_text(data.text2)
        # Preprocesses both input texts using the 'preprocess_text' function.

        # Inference
        embeddings = model.encode(
            [text1, text2],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        # Encodes the preprocessed texts into embeddings using the SentenceTransformer model:
        # - '[text1, text2]': Encodes both texts in a single batch.
        # - 'convert_to_tensor=True': Returns PyTorch tensors for efficient computation.
        # - 'normalize_embeddings=True': Normalizes embeddings to unit length for cosine similarity.

        # Calculate similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        similarity = max(0.0, min(1.0, similarity))
        # Computes the cosine similarity between the two embeddings:
        # - 'util.cos_sim': Calculates the cosine similarity score.
        # - '.item()': Converts the tensor result to a Python float.
        # - Clamps the similarity score to the range [0, 1].

        return {"similarity score": similarity}
        # Returns the similarity score in the format defined by 'SimilarityResponse'.

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing request"
        )
        # Catches any errors during processing, logs them, and raises a 500 error.

@app.get("/health")
def health_check():
    return {
        "status": "ready" if model else "loading",
        "model_loaded": model is not None
    }
    # Defines a health check endpoint at '/health':
    # - Returns a JSON object indicating the API's status.
    # - 'status': "ready" if the model is loaded, "loading" if not.
    # - 'model_loaded': Boolean indicating whether the model is loaded.

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        log_config=None
    )
    # Runs the FastAPI application using Uvicorn if the script is executed directly:
    # - 'app': The FastAPI application instance.
    # - 'host="0.0.0.0"': Binds the server to all network interfaces.
    # - 'port': Uses the 'PORT' environment variable or defaults to 8080.
    # - 'log_config=None': Disables Uvicorn's default logging to use the custom logging configuration.
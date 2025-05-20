# DataNeuron Text Similarity API
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from typing import List
from pathlib import Path

# Configure logging for traceability and debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TextSimilarityModel:
    """Class to compute semantic similarity between text pairs using Sentence-BERT."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the model with a Sentence-BERT variant.
        
        Args:
            model_name (str): Name of the Sentence-BERT model (default: all-MiniLM-L6-v2).
        """
        try:
            logger.info(f"Loading Sentence-BERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocess text: lowercase, remove special characters, strip whitespace.
        
        Args:
            text (str): Input text to preprocess.
        
        Returns:
            str: Preprocessed text.
        """
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to encode.
        
        Returns:
            np.ndarray: Array of embeddings.
        """
        try:
            logger.info(f"Computing embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings, normalized to [0, 1].
        
        Args:
            embedding1 (np.ndarray): First text embedding.
            embedding2 (np.ndarray): Second text embedding.
        
        Returns:
            float: Normalized similarity score (0–1).
        """
        try:
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            # Normalize to [0, 1]
            normalized_similarity = (similarity + 1) / 2
            return np.clip(normalized_similarity, 0, 1)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise

def load_and_preprocess_data(file_path: str, model: TextSimilarityModel) -> pd.DataFrame:
    """Load CSV and preprocess text columns.
    
    Args:
        file_path (str): Path to input CSV.
        model (TextSimilarityModel): Instance of the similarity model for preprocessing.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(df)} rows")

        # Preprocess text columns
        df['text1'] = df['text1'].apply(model.preprocess_text)
        df['text2'] = df['text2'].apply(model.preprocess_text)

        # Remove invalid rows
        initial_rows = len(df)
        df = df.dropna(subset=['text1', 'text2'])
        df = df[df['text1'].str.strip() != '']
        df = df[df['text2'].str.strip() != '']
        logger.info(f"After preprocessing, {len(df)} rows remain (removed {initial_rows - len(df)})")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def compute_similarity_scores(df: pd.DataFrame, model: TextSimilarityModel) -> List[float]:
    """Compute similarity scores for text pairs in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with preprocessed text pairs.
        model (TextSimilarityModel): Instance of the similarity model.
    
    Returns:
        List[float]: List of similarity scores.
    """
    try:
        logger.info("Computing embeddings for text pairs")
        text1_embeddings = model.compute_embeddings(df['text1'].tolist())
        text2_embeddings = model.compute_embeddings(df['text2'].tolist())

        logger.info("Calculating similarity scores")
        similarity_scores = [
            model.calculate_similarity(emb1, emb2)
            for emb1, emb2 in zip(text1_embeddings, text2_embeddings)
        ]
        return similarity_scores
    except Exception as e:
        logger.error(f"Error computing similarity scores: {e}")
        raise

def save_results(df: pd.DataFrame, output_path: str) -> None:
    """Save dataframe with similarity scores to CSV.
    
    Args:
        df (pd.DataFrame): Dataframe with text pairs and similarity scores.
        output_path (str): Path to output CSV.
    """
    try:
        logger.info(f"Saving results to {output_path}")
        df[['text1', 'text2', 'similarity_score']].to_csv(output_path, index=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    """Main function to execute the similarity computation pipeline."""
    try:
        # Initialize model
        similarity_model = TextSimilarityModel()

        # Define paths
        input_path = Path("DataNeuron_Text_Similarity.csv")
        output_path = Path("similarity_scores.csv")

        # Load and preprocess data
        df = load_and_preprocess_data(input_path, similarity_model)

        # Compute similarity scores
        df['similarity_score'] = compute_similarity_scores(df, similarity_model)

        # Save results
        save_results(df, output_path)

        # Log sample results
        logger.info("Sample results:")
        for _, row in df.head(5).iterrows():
            logger.info(
                f"Text1: {row['text1'][:50]}...\n"
                f"Text2: {row['text2'][:50]}...\n"
                f"Similarity Score: {row['similarity_score']:.4f}\n"
            )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
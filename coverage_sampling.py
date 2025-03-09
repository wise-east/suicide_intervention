import pandas as pd
from pathlib import Path 
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
from loguru import logger
from typing import List


def load_data(full_data_path, sample_data_path, sample_size=1000):
    """Load data from file or create and save a sample if it doesn't exist."""
    if sample_data_path.exists():
        logger.info(f"Loading existing sample data from {sample_data_path}")
        return pd.read_csv(sample_data_path)
    else:
        logger.info(f"Creating new sample from {full_data_path}")
        df = pd.read_stata(full_data_path)
        df_sample = df.sample(n=sample_size)
        df_sample.to_csv(sample_data_path, index=False)
        return df_sample


def prepare_narratives(df):
    """Combine narrative columns and tokenize into sentences."""
    df['Narrative'] = df['NarrativeLE'] + ' ' + df['NarrativeCME']
    all_narratives = df['Narrative'].tolist()
    all_narratives_sentences = [sent_tokenize(narrative) for narrative in all_narratives]
    
    # Flatten the list of sentences
    flattened_sentences = [sentence for narrative in all_narratives_sentences for sentence in narrative]
    logger.info(f'Sentence tokenized {len(flattened_sentences)} sentences')
    
    return all_narratives_sentences, flattened_sentences


def embed_sentences(sentences, model_name='all-MiniLM-L6-v2'):
    """Embed sentences using a sentence transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    logger.info(f'Embedded {len(embeddings)} sentences')
    
    return embeddings


def compute_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix for all embeddings using vectorized operations."""
    # Convert to numpy array if not already
    embeddings = np.array(embeddings)
    
    # Normalize the vectors for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Use dot product of normalized vectors (much faster than pairwise)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    logger.info(f'Cosine similarity matrix computed with shape {similarity_matrix.shape}')
    return similarity_matrix


# Alternative using batch processing for very large matrices
def compute_similarity_matrix_batched(embeddings, batch_size=1000):
    """Compute similarity matrix in batches to avoid memory issues."""
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    # Normalize embeddings first (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        batch = normalized_embeddings[i:end_idx]
        
        # Compute similarity for this batch against all embeddings
        batch_similarities = np.dot(batch, normalized_embeddings.T)
        similarity_matrix[i:end_idx] = batch_similarities
    
    logger.info(f'Cosine similarity matrix computed with batched processing')
    return similarity_matrix


def compute_average_max_similarity(chosen_indices, narrative_indices, similarity_matrix):
    """Compute average max cosine similarity between chosen and candidate narratives."""
    max_similarities = []
    
    for sentence_idx in narrative_indices:
        similarities = [similarity_matrix[sentence_idx, chosen_idx] for chosen_idx in chosen_indices]
        max_similarities.append(max(similarities) if similarities else 0)
    
    return np.mean(max_similarities) if max_similarities else 0


def select_narratives_based_on_coverage(all_narratives_sentences: List[List[str]], flattened_sentences: List[str], 
                            similarity_matrix, sample_size=5, max_iterations=10):
    """Select diverse narratives using a greedy algorithm based on similarity."""
    # Create sentence to index lookup
    sentence_to_index = {sentence: i for i, sentence in enumerate(flattened_sentences)}
    
    # Copy to avoid modifying the original
    remaining_narratives = all_narratives_sentences.copy()
    chosen_narratives = []
    
    for _ in range(max_iterations):
        # When chosen narratives is empty, randomly select one
        if not chosen_narratives:
            idx = np.random.randint(0, len(remaining_narratives))
            chosen_narratives.append(remaining_narratives.pop(idx))
        else:
            # Get indices of all sentences in chosen narratives
            chosen_sentences = [sentence for narrative in chosen_narratives for sentence in narrative]
            chosen_indices = [sentence_to_index[sentence] for sentence in chosen_sentences]
            
            # Calculate similarity for each remaining narrative
            similarities = []
            for narrative in remaining_narratives:
                narrative_indices = [sentence_to_index[sentence] for sentence in narrative]
                similarity = compute_average_max_similarity(chosen_indices, narrative_indices, similarity_matrix)
                similarities.append(similarity)
            
            # Select the most diverse narratives (lowest similarity)
            bottom_indices = np.argsort(similarities)[:sample_size]
            
            # Add the selected narratives to chosen_narratives
            for idx in sorted(bottom_indices, reverse=True):
                if idx < len(remaining_narratives):  # Safety check
                    chosen_narratives.append(remaining_narratives.pop(idx))
    
    return chosen_narratives


def main():
    # Define paths
    data_path = 'data'
    full_data_fn = Path(data_path) / 'clark_492_nvdrs_2019.dta'
    sample_data_fn = Path(data_path) / 'lawyer_attorney_faiss.csv'
    
    # Load data
    df_sample = load_data(full_data_fn, sample_data_fn)
    
    # use only 100 rows for testing
    df_sample = df_sample.head(30)
    
    # Prepare narratives
    all_narratives_sentences, flattened_sentences = prepare_narratives(df_sample)
    
    # Embed sentences
    sentence_embeddings = embed_sentences(flattened_sentences)
    
    # Compute similarity matrix - choose the appropriate method based on your data size
    if len(sentence_embeddings) > 10000:
        similarity_matrix = compute_similarity_matrix_batched(sentence_embeddings)
    else:
        similarity_matrix = compute_similarity_matrix(sentence_embeddings)
    
    # Select diverse narratives
    chosen_narratives = select_narratives_based_on_coverage(
        all_narratives_sentences, 
        flattened_sentences,
        similarity_matrix, 
        sample_size=1, 
        max_iterations=10
    )
    
    logger.info(f"Selected {len(chosen_narratives)} diverse narratives")
    
    return chosen_narratives


if __name__ == "__main__":
    main()


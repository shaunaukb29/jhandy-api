import numpy as np
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "_fixtures"

def load_reference_corpus() -> tuple[np.ndarray, list[dict]]:
    """Load the pre-computed fixture embeddings and metadata."""
    npz = FIXTURES / "embeddings.npz"
    if not npz.exists():
        # Fallback if fixtures don't exist
        return np.zeros((0, 512)), []
    
    z = np.load(npz, allow_pickle=False)
    X = z["X"]
    
    def _clean(s):
        s = str(s)
        return "" if s.lower() in ("none", "nan") else s
        
    metadata = [
        {
            "clip_id": str(c),
            "video": str(v),
            "kind": _clean(k),
            "l1": _clean(l1v),
            "cause": _clean(ca)
        }
        for c, v, k, l1v, ca in zip(z["clip_id"], z["video"], z["kind"], z["l1"], z["cause"])
    ]
    return X, metadata

def find_similar_audio(query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
    """Find the top_k most similar reference audio clips to query_vector."""
    X, metadata = load_reference_corpus()
    if X.shape[0] == 0:
        return []
        
    # Ensure query_vector is 1D or a single row
    if query_vector.ndim > 1:
        query_vector = query_vector.flatten()
        
    # Normalize query vector if it is not already L2-normalized
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm
        
    # Compute cosine similarities via dot product (X is already L2-normalized)
    similarities = np.dot(X, query_vector)
    
    # Get top_k indices sorted descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        meta = metadata[idx]
        sim = float(similarities[idx])
        results.append({
            "clip_id": meta["clip_id"],
            "video": meta["video"],
            "kind": meta["kind"],
            "l1": meta["l1"],
            "cause": meta["cause"],
            "similarity": sim
        })
    return results

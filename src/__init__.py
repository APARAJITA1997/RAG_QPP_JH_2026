# src/__init__.py
from .features    import extract_features, extract_features_batch, normalize_features, FEATURE_NAMES
from .retrieval   import BM25Retriever, DenseRetriever, HybridRetriever, ColBERTRetriever, build_retriever
from .models      import train_model, train_all_models, predict, save_model, load_model
from .generation  import BARTGenerator, LLaMAGenerator, build_generator
from .evaluate    import compute_correlations, compute_retrieval_metrics, compute_generation_metrics, stratified_analysis
from .adaptive_rag import AdaptiveRAGPipeline

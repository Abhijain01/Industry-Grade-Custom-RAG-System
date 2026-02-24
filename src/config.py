import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class SystemConfig:
    log_level: str
    log_file: str
    temp_dir: str

@dataclass
class IngestionConfig:
    chunk_size: int
    chunk_overlap: int
    max_workers: int
    supported_extensions: List[str]

@dataclass
class EmbeddingConfig:
    model_name: str
    batch_size: int
    device: str

@dataclass
class VectorStoreConfig:
    index_path: str
    metadata_path: str
    metric: str

@dataclass
class RetrievalConfig:
    top_k: int
    mmr_diversity: float
    enable_bm25: bool

@dataclass
class GenerationConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int

@dataclass
class AppConfig:
    system: SystemConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

def load_config(config_path: str = "config.yaml") -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(
        system=SystemConfig(**data["system"]),
        ingestion=IngestionConfig(**data["ingestion"]),
        embedding=EmbeddingConfig(**data["embedding"]),
        vector_store=VectorStoreConfig(**data["vector_store"]),
        retrieval=RetrievalConfig(**data["retrieval"]),
        generation=GenerationConfig(**data["generation"])
    )

# Global configuration instance
CONFIG: Optional[AppConfig] = None
try:
    CONFIG = load_config(os.path.join(Path(__file__).resolve().parent.parent, "config.yaml"))
except Exception as e:
    pass # Will be re-loaded manually in tests

"""Inference-only OPERA package."""

from .pipeline import OperaPipeline, PipelineConfig
from .retriever import BGEM3FaissRetriever

__all__ = ["OperaPipeline", "PipelineConfig", "BGEM3FaissRetriever"]

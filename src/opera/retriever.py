from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import Document


class BGEM3FaissRetriever:
    """BGE-M3 dense retriever over the prebuilt OPERA FAISS index."""

    def __init__(
        self,
        *,
        index_path: str,
        metadata_path: str,
        bge_model: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        faiss_gpu: bool = True,
        gpu_id: int = 0,
        strict_count_check: bool = True,
    ):
        import faiss
        import numpy as np
        from FlagEmbedding import BGEM3FlagModel

        self.faiss = faiss
        self.np = np
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.gpu_id = gpu_id
        self._gpu_resources = None

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata not found: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("rb") as f:
            self.metadata = pickle.load(f)
        if isinstance(self.metadata, dict) and "documents" in self.metadata:
            self.metadata = self.metadata["documents"]

        if strict_count_check and self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"index/doc count mismatch: index={self.index.ntotal}, metadata={len(self.metadata)}"
            )

        if faiss_gpu:
            if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
                raise RuntimeError("faiss_gpu=True but FAISS reports no GPUs")
            self._gpu_resources = faiss.StandardGpuResources()
            try:
                options = faiss.GpuClonerOptions()
                options.useFloat16 = True
                self.index = faiss.index_cpu_to_gpu(
                    self._gpu_resources,
                    gpu_id,
                    self.index,
                    options,
                )
            except TypeError:
                self.index = faiss.index_cpu_to_gpu(self._gpu_resources, gpu_id, self.index)

        self.embedding_model = BGEM3FlagModel(bge_model, use_fp16=use_fp16)

    def _encode(self, query: str):
        try:
            encoded = self.embedding_model.encode(
                [query],
                batch_size=1,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            encoded = self.embedding_model.encode([query], batch_size=1)

        vectors = encoded["dense_vecs"] if isinstance(encoded, dict) else encoded
        vectors = self.np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-12)
        return vectors.astype("float32")

    @staticmethod
    def _doc_text(raw: Dict[str, Any]) -> str:
        return (
            raw.get("content")
            or raw.get("contents")
            or raw.get("paragraph_text")
            or raw.get("text")
            or ""
        )

    def search(self, query: str, *, top_k: int = 5) -> List[Document]:
        query_vec = self._encode(query)
        scores, indices = self.index.search(query_vec, top_k)
        results: List[Document] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            raw = self.metadata[int(idx)]
            doc_id = str(raw.get("id") or raw.get("doc_id") or idx)
            metadata = dict(raw.get("metadata") or {})
            if raw.get("title") and "title" not in metadata:
                metadata["title"] = raw.get("title")
            results.append(
                Document(
                    doc_id=doc_id,
                    title=str(raw.get("title") or metadata.get("title") or ""),
                    content=self._doc_text(raw),
                    score=float(score),
                    rank=rank,
                    metadata=metadata,
                )
            )
        return results

    def info(self) -> Dict[str, Any]:
        return {
            "backend": "local_bge_m3_faiss",
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "index_size": int(self.index.ntotal),
            "metadata_size": len(self.metadata),
            "gpu_id": self.gpu_id,
        }


class BGEM3MultiFaissRetriever:
    """BGE-M3 retriever with one encoder shared by multiple FAISS indexes."""

    def __init__(
        self,
        *,
        datasets: Dict[str, Dict[str, str]],
        bge_model: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        faiss_gpu: bool = True,
        gpu_id: int = 0,
        strict_count_check: bool = True,
    ):
        import faiss
        import numpy as np
        from FlagEmbedding import BGEM3FlagModel

        if not datasets:
            raise ValueError("datasets must not be empty")

        self.faiss = faiss
        self.np = np
        self.gpu_id = gpu_id
        self._gpu_resources = None
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.default_dataset = next(iter(datasets))

        if faiss_gpu:
            if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
                raise RuntimeError("faiss_gpu=True but FAISS reports no GPUs")
            self._gpu_resources = faiss.StandardGpuResources()

        for name, spec in datasets.items():
            index_path = Path(spec["index_path"])
            metadata_path = Path(spec["metadata_path"])
            if not index_path.exists():
                raise FileNotFoundError(f"{name}: FAISS index not found: {index_path}")
            if not metadata_path.exists():
                raise FileNotFoundError(f"{name}: metadata not found: {metadata_path}")

            index = faiss.read_index(str(index_path))
            with metadata_path.open("rb") as f:
                metadata = pickle.load(f)
            if isinstance(metadata, dict) and "documents" in metadata:
                metadata = metadata["documents"]

            if strict_count_check and index.ntotal != len(metadata):
                raise ValueError(
                    f"{name}: index/doc count mismatch: index={index.ntotal}, metadata={len(metadata)}"
                )

            if faiss_gpu:
                try:
                    options = faiss.GpuClonerOptions()
                    options.useFloat16 = True
                    index = faiss.index_cpu_to_gpu(self._gpu_resources, gpu_id, index, options)
                except TypeError:
                    index = faiss.index_cpu_to_gpu(self._gpu_resources, gpu_id, index)

            self.datasets[name] = {
                "index": index,
                "metadata": metadata,
                "index_path": index_path,
                "metadata_path": metadata_path,
            }

        self.embedding_model = BGEM3FlagModel(bge_model, use_fp16=use_fp16)

    def _encode(self, query: str):
        try:
            encoded = self.embedding_model.encode(
                [query],
                batch_size=1,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            encoded = self.embedding_model.encode([query], batch_size=1)

        vectors = encoded["dense_vecs"] if isinstance(encoded, dict) else encoded
        vectors = self.np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-12)
        return vectors.astype("float32")

    def search(self, query: str, *, top_k: int = 5, dataset: Optional[str] = None) -> List[Document]:
        dataset_name = dataset or self.default_dataset
        if dataset_name not in self.datasets:
            available = ", ".join(sorted(self.datasets))
            raise KeyError(f"unknown dataset {dataset_name!r}; available: {available}")

        item = self.datasets[dataset_name]
        query_vec = self._encode(query)
        scores, indices = item["index"].search(query_vec, top_k)
        metadata_list = item["metadata"]

        results: List[Document] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            raw = metadata_list[int(idx)]
            doc_id = str(raw.get("id") or raw.get("doc_id") or idx)
            metadata = dict(raw.get("metadata") or {})
            metadata["dataset"] = dataset_name
            if raw.get("title") and "title" not in metadata:
                metadata["title"] = raw.get("title")
            results.append(
                Document(
                    doc_id=doc_id,
                    title=str(raw.get("title") or metadata.get("title") or ""),
                    content=BGEM3FaissRetriever._doc_text(raw),
                    score=float(score),
                    rank=rank,
                    metadata=metadata,
                )
            )
        return results

    def info(self) -> Dict[str, Any]:
        return {
            "backend": "local_bge_m3_multi_faiss",
            "default_dataset": self.default_dataset,
            "datasets": {
                name: {
                    "index_path": str(item["index_path"]),
                    "metadata_path": str(item["metadata_path"]),
                    "index_size": int(item["index"].ntotal),
                    "metadata_size": len(item["metadata"]),
                }
                for name, item in self.datasets.items()
            },
            "gpu_id": self.gpu_id,
        }


class HTTPRetrieverClient:
    """Client for a persistent OPERA retriever server."""

    def __init__(self, base_url: str, *, timeout: int = 180, dataset: Optional[str] = None):
        import requests

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dataset = dataset
        self.session = requests.Session()

    def set_dataset(self, dataset: Optional[str]) -> None:
        self.dataset = dataset

    def search(self, query: str, *, top_k: int = 5) -> List[Document]:
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if self.dataset:
            payload["dataset"] = self.dataset
        resp = self.session.post(
            f"{self.base_url}/search",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            Document(
                doc_id=str(item.get("doc_id", "")),
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                score=float(item.get("score", 0.0)),
                rank=int(item.get("rank", idx)),
                metadata=dict(item.get("metadata") or {}),
            )
            for idx, item in enumerate(data.get("documents", []), start=1)
        ]

    def info(self) -> Dict[str, Any]:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            data = {"error": str(exc)}
        return {
            "backend": "http_retriever",
            "base_url": self.base_url,
            "dataset": self.dataset,
            **data,
        }

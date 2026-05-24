from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from .cli import DEFAULT_BGE_LOCAL, DEFAULT_INDEX, DEFAULT_META
from .retriever import BGEM3FaissRetriever, BGEM3MultiFaissRetriever
from .utils import force_gpu0


class RetrieverService:
    def __init__(self, retriever: BGEM3FaissRetriever | BGEM3MultiFaissRetriever, *, max_top_k: int = 50):
        self.retriever = retriever
        self.max_top_k = max_top_k
        self.started_at = time.time()
        self.lock = threading.RLock()
        self.num_queries = 0

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "opera_retriever_server",
            "uptime_sec": time.time() - self.started_at,
            "num_queries": self.num_queries,
            "max_top_k": self.max_top_k,
            "retriever": self.retriever.info(),
        }

    def search(self, query: str, top_k: int, dataset: str | None = None) -> Dict[str, Any]:
        top_k = max(1, min(int(top_k), self.max_top_k))
        with self.lock:
            if dataset and hasattr(self.retriever, "datasets"):
                docs = self.retriever.search(query, top_k=top_k, dataset=dataset)
            else:
                docs = self.retriever.search(query, top_k=top_k)
            self.num_queries += 1
        return {
            "query": query,
            "top_k": top_k,
            "dataset": dataset,
            "documents": [doc.to_dict() for doc in docs],
        }


def make_handler(service: RetrieverService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OperaRetrieverHTTP/0.1"

        def _send_json(self, payload: Dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(service.health())
                return
            if parsed.path == "/search":
                params = parse_qs(parsed.query)
                query = (params.get("query") or params.get("q") or [""])[0]
                top_k = int((params.get("top_k") or ["5"])[0])
                dataset = (params.get("dataset") or [""])[0] or None
                if not query.strip():
                    self._send_json({"error": "missing query"}, status=400)
                    return
                self._send_json(service.search(query, top_k, dataset=dataset))
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/search":
                    query = str(payload.get("query") or payload.get("q") or "")
                    top_k = int(payload.get("top_k") or 5)
                    dataset = str(payload.get("dataset") or "") or None
                    if not query.strip():
                        self._send_json({"error": "missing query"}, status=400)
                        return
                    self._send_json(service.search(query, top_k, dataset=dataset))
                    return
                if parsed.path == "/batch_search":
                    queries = payload.get("queries") or []
                    top_k = int(payload.get("top_k") or 5)
                    dataset = str(payload.get("dataset") or "") or None
                    if not isinstance(queries, list):
                        self._send_json({"error": "queries must be a list"}, status=400)
                        return
                    results: List[Dict[str, Any]] = []
                    for query in queries:
                        results.append(service.search(str(query), top_k, dataset=dataset))
                    self._send_json({"results": results})
                    return
                self._send_json({"error": "not found"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent OPERA BGE-M3 + FAISS retriever server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--index-path", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_META)
    parser.add_argument(
        "--dataset-index",
        action="append",
        default=[],
        help="Dataset-specific index in name=/path/index.faiss:/path/meta.pkl form. Can be repeated.",
    )
    parser.add_argument("--bge-model", type=str, default=DEFAULT_BGE_LOCAL)
    parser.add_argument("--max-top-k", type=int, default=50)
    parser.add_argument("--no-faiss-gpu", action="store_true")
    parser.add_argument("--bge-fp32", action="store_true")
    return parser


def parse_dataset_indexes(specs: List[str]) -> Dict[str, Dict[str, str]]:
    datasets: Dict[str, Dict[str, str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --dataset-index value: {spec}. Use name=/path/index:/path/meta")
        name, payload = spec.split("=", 1)
        if ":" not in payload:
            raise SystemExit(f"Invalid --dataset-index value: {spec}. Use name=/path/index:/path/meta")
        index_path, metadata_path = payload.split(":", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid --dataset-index value: {spec}. Dataset name is empty")
        datasets[name] = {
            "index_path": index_path.strip(),
            "metadata_path": metadata_path.strip(),
        }
    return datasets


def main() -> None:
    args = build_arg_parser().parse_args()
    force_gpu0(args.gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_indexes = parse_dataset_indexes(args.dataset_index)
    if dataset_indexes:
        print(
            json.dumps(
                {
                    "event": "loading_multi_retriever",
                    "gpu_id": args.gpu_id,
                    "datasets": dataset_indexes,
                    "bge_model": args.bge_model,
                    "faiss_gpu": not args.no_faiss_gpu,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        retriever = BGEM3MultiFaissRetriever(
            datasets=dataset_indexes,
            bge_model=args.bge_model,
            use_fp16=not args.bge_fp32,
            faiss_gpu=not args.no_faiss_gpu,
            gpu_id=0,
        )
    else:
        print(
            json.dumps(
                {
                    "event": "loading_retriever",
                    "gpu_id": args.gpu_id,
                    "index_path": args.index_path,
                    "metadata_path": args.metadata_path,
                    "bge_model": args.bge_model,
                    "faiss_gpu": not args.no_faiss_gpu,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        retriever = BGEM3FaissRetriever(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            bge_model=args.bge_model,
            use_fp16=not args.bge_fp32,
            faiss_gpu=not args.no_faiss_gpu,
            gpu_id=0,
        )
    service = RetrieverService(retriever, max_top_k=args.max_top_k)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(json.dumps({"event": "server_ready", "host": args.host, "port": args.port, **service.health()}, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

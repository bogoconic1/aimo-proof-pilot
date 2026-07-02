from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd


ZEMBED_QUERY_PROMPT = "<|im_start|>system\nquery<|im_end|>\n<|im_start|>user\n"
ZEMBED_DOC_PROMPT = "<|im_start|>system\ndocument<|im_end|>\n<|im_start|>user\n"
ZEMBED_SUFFIX = "<|im_end|>\n"

DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_RERANK_BASE_URL = "http://127.0.0.1:20129"
DEFAULT_API_KEY = "vllm-local"
DEFAULT_EMBED_MODEL = "zembed-1"
DEFAULT_RERANK_MODEL = "zerank-2"
DEFAULT_DIMENSIONS = 2560


Backend = Literal["vllm", "zeroentropy"]


@dataclass
class RagConfig:
    index_path: Path
    retrieve_top_k: int = 100
    rerank_top_k: int = 5
    embed_backend: Backend = "vllm"
    embed_base_url: str = DEFAULT_EMBED_BASE_URL
    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_backend: Backend = "vllm"
    rerank_base_url: str = DEFAULT_RERANK_BASE_URL
    rerank_model: str = DEFAULT_RERANK_MODEL
    max_example_solution_chars: int = 6000
    api_key: str = DEFAULT_API_KEY
    dimensions: int = DEFAULT_DIMENSIONS
    timeout_s: int = 120


@dataclass
class RagExample:
    row_id: int
    question: str
    solution: str
    rag_text: str
    retrieve_score: float
    rerank_score: Optional[float] = None


def clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def build_rag_text(question: Any, solution: Any) -> str:
    return f"## Problem\n{clean_cell_text(question)}\n\n## Solution\n{clean_cell_text(solution)}"


def format_zembed_query(text: str) -> str:
    return f"{ZEMBED_QUERY_PROMPT}{str(text or '').strip()}{ZEMBED_SUFFIX}"


def format_zembed_document(text: str) -> str:
    return f"{ZEMBED_DOC_PROMPT}{str(text or '').strip()}{ZEMBED_SUFFIX}"


def l2_normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        denom = float(np.linalg.norm(array))
        if denom == 0.0:
            return array.astype(np.float32)
        return (array / denom).astype(np.float32)
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return (array / denom).astype(np.float32)


def decode_embedding(value: object, encoding_format: str = "float") -> list[float]:
    if encoding_format == "float":
        return [float(item) for item in value]  # type: ignore[union-attr]
    raw = base64.b64decode(str(value))
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace") if error.fp else ""
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail[:500]}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Request failed for {url}: {error}") from error


def embed_texts_vllm(
    texts: list[str],
    *,
    base_url: str,
    model: str,
    api_key: str,
    dimensions: int = 0,
    timeout_s: int = 120,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions > 0:
        payload["dimensions"] = dimensions
    body = _post_json(
        f"{base_url.rstrip('/')}/embeddings",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    data = body.get("data", [])
    if len(data) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(data)}")
    return np.array([item["embedding"] for item in data], dtype=np.float32)


def embed_texts_zeroentropy(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    dimensions: int,
    input_type: Literal["query", "document"],
    timeout_s: int = 120,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    payload = {
        "model": model,
        "input_type": input_type,
        "input": texts,
        "dimensions": dimensions,
        "encoding_format": "float",
    }
    body = _post_json(
        "https://api.zeroentropy.dev/v1/models/embed",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    results = body.get("results", [])
    if len(results) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(results)}")
    return np.array([decode_embedding(item["embedding"]) for item in results], dtype=np.float32)


def embed_texts(
    texts: list[str],
    *,
    backend: Backend,
    base_url: str,
    model: str,
    api_key: str,
    dimensions: int,
    input_type: Literal["query", "document"],
    timeout_s: int,
) -> np.ndarray:
    if backend == "vllm":
        formatted = [
            format_zembed_query(text) if input_type == "query" else format_zembed_document(text)
            for text in texts
        ]
        return embed_texts_vllm(
            formatted,
            base_url=base_url,
            model=model,
            api_key=api_key,
            dimensions=0 if dimensions == DEFAULT_DIMENSIONS else dimensions,
            timeout_s=timeout_s,
        )
    if backend == "zeroentropy":
        return embed_texts_zeroentropy(
            texts,
            model=model,
            api_key=api_key or os.environ.get("ZEROENTROPY_API_KEY", ""),
            dimensions=dimensions,
            input_type=input_type,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")


def rerank_documents_vllm(
    query: str,
    documents: list[str],
    *,
    base_url: str,
    model: str,
    api_key: str,
    top_n: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    if not documents:
        return []
    body = _post_json(
        f"{base_url.rstrip('/').removesuffix('/v1')}/rerank",
        {"model": model, "query": query, "documents": documents, "top_n": top_n},
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout_s,
    )
    return [
        {"index": int(item["index"]), "relevance_score": float(item["relevance_score"])}
        for item in body.get("results", [])
    ]


def rerank_documents_zeroentropy(
    query: str,
    documents: list[str],
    *,
    model: str,
    api_key: str,
    top_n: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    if not documents:
        return []
    body = _post_json(
        "https://api.zeroentropy.dev/v1/models/rerank",
        {"model": model, "query": query, "documents": documents, "top_n": top_n},
        {
            "Authorization": f"Bearer {api_key or os.environ.get('ZEROENTROPY_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        timeout_s,
    )
    return [
        {"index": int(item["index"]), "relevance_score": float(item["relevance_score"])}
        for item in body.get("results", [])
    ]


def rerank_documents(
    query: str,
    documents: list[str],
    *,
    backend: Backend,
    base_url: str,
    model: str,
    api_key: str,
    top_n: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    if backend == "vllm":
        return rerank_documents_vllm(
            query,
            documents,
            base_url=base_url,
            model=model,
            api_key=api_key,
            top_n=top_n,
            timeout_s=timeout_s,
        )
    if backend == "zeroentropy":
        return rerank_documents_zeroentropy(
            query,
            documents,
            model=model,
            api_key=api_key,
            top_n=top_n,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unsupported rerank backend: {backend}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported table format: {path}")


def build_math_rag_index(
    *,
    input_path: Path,
    output_path: Path,
    question_column: str = "question",
    solution_column: str = "solution",
    batch_size: int = 32,
    backend: Backend = "vllm",
    base_url: str = DEFAULT_EMBED_BASE_URL,
    model: str = DEFAULT_EMBED_MODEL,
    api_key: str = DEFAULT_API_KEY,
    dimensions: int = DEFAULT_DIMENSIONS,
    timeout_s: int = 120,
    embed_fn: Optional[Any] = None,
) -> pd.DataFrame:
    df = read_table(input_path)
    for column in (question_column, solution_column):
        if column not in df.columns:
            raise ValueError(f"{input_path} is missing required column {column!r}")
    rows: list[dict[str, Any]] = []
    for row_id, row in df.reset_index(drop=True).iterrows():
        question = clean_cell_text(row[question_column])
        solution = clean_cell_text(row[solution_column])
        rows.append(
            {
                "row_id": int(row_id),
                "question": question,
                "solution": solution,
                "rag_text": build_rag_text(question, solution),
            }
        )
    embeddings: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [row["rag_text"] for row in batch]
        if embed_fn is not None:
            vectors = np.array(embed_fn(texts), dtype=np.float32)
        else:
            vectors = embed_texts(
                texts,
                backend=backend,
                base_url=base_url,
                model=model,
                api_key=api_key,
                dimensions=dimensions,
                input_type="document",
                timeout_s=timeout_s,
            )
        embeddings.append(l2_normalize_array(vectors))
        done = min(start + batch_size, len(rows))
        print(f"Embedded {done}/{len(rows)} RAG rows", end="\r", flush=True)
    print()
    matrix = np.vstack(embeddings) if embeddings else np.empty((0, dimensions), dtype=np.float32)
    out_df = pd.DataFrame(rows)
    out_df["embedding"] = [matrix[idx].astype(np.float32).tolist() for idx in range(len(out_df))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    return out_df


class MathRagRetriever:
    def __init__(
        self,
        config: RagConfig,
        *,
        embed_fn: Optional[Any] = None,
        rerank_fn: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        df = pd.read_parquet(config.index_path)
        required = {"row_id", "question", "solution", "rag_text", "embedding"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{config.index_path} missing RAG columns: {sorted(missing)}")
        self.rows = df.reset_index(drop=True)
        matrix = np.vstack([
            np.array(value, dtype=np.float32)
            for value in self.rows["embedding"].tolist()
        ])
        self.embeddings = l2_normalize_array(matrix)

    def embed_query(self, question: str) -> np.ndarray:
        if self._embed_fn is not None:
            vector = self._embed_fn(question)
            return l2_normalize_array(np.array(vector, dtype=np.float32))
        vectors = embed_texts(
            [question],
            backend=self.config.embed_backend,
            base_url=self.config.embed_base_url,
            model=self.config.embed_model,
            api_key=self.config.api_key,
            dimensions=self.config.dimensions,
            input_type="query",
            timeout_s=self.config.timeout_s,
        )
        return l2_normalize_array(vectors[0])

    def rerank(self, question: str, documents: list[str]) -> list[dict[str, Any]]:
        if self._rerank_fn is not None:
            return self._rerank_fn(question, documents)
        return rerank_documents(
            question,
            documents,
            backend=self.config.rerank_backend,
            base_url=self.config.rerank_base_url,
            model=self.config.rerank_model,
            api_key=self.config.api_key,
            top_n=min(self.config.rerank_top_k, len(documents)),
            timeout_s=self.config.timeout_s,
        )

    def retrieve(self, question: str) -> list[RagExample]:
        if len(self.rows) == 0:
            return []
        query_vector = self.embed_query(question)
        scores = self.embeddings @ query_vector
        retrieve_count = max(1, min(int(self.config.retrieve_top_k), len(scores)))
        candidate_indices = np.argpartition(-scores, retrieve_count - 1)[:retrieve_count]
        candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        candidate_docs = [str(self.rows.iloc[int(idx)]["rag_text"]) for idx in candidate_indices]
        reranked = self.rerank(question, candidate_docs)
        if not reranked:
            reranked = [
                {"index": rank, "relevance_score": float(scores[int(idx)])}
                for rank, idx in enumerate(candidate_indices[: self.config.rerank_top_k])
            ]
        examples: list[RagExample] = []
        for item in reranked[: self.config.rerank_top_k]:
            local_index = int(item["index"])
            if not (0 <= local_index < len(candidate_indices)):
                continue
            source_index = int(candidate_indices[local_index])
            row = self.rows.iloc[source_index]
            examples.append(
                RagExample(
                    row_id=int(row["row_id"]),
                    question=str(row["question"]),
                    solution=str(row["solution"]),
                    rag_text=str(row["rag_text"]),
                    retrieve_score=float(scores[source_index]),
                    rerank_score=float(item.get("relevance_score", 0.0)),
                )
            )
        return examples


def format_rag_examples_for_prompt(
    examples: list[RagExample],
    *,
    max_solution_chars: int,
) -> str:
    if not examples:
        return ""
    blocks = [
        "## Retrieved Similar Solved Examples",
        "These examples may contain useful proof patterns. Do not copy an argument unless it applies rigorously.",
    ]
    for idx, example in enumerate(examples, start=1):
        solution = example.solution.strip()
        if max_solution_chars > 0 and len(solution) > max_solution_chars:
            solution = solution[:max_solution_chars].rstrip() + "\n...[truncated]"
        blocks.append(
            f"### Example {idx}\n"
            f"Problem:\n{example.question.strip()}\n\n"
            f"Solution:\n{solution}"
        )
    return "\n\n".join(blocks)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a math proof RAG embedding index.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--solution-column", default="solution")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", choices=["vllm", "zeroentropy"], default="vllm")
    parser.add_argument("--base-url", default=DEFAULT_EMBED_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--timeout-s", type=int, default=120)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_math_rag_index(
        input_path=args.input,
        output_path=args.output,
        question_column=args.question_column,
        solution_column=args.solution_column,
        batch_size=max(1, args.batch_size),
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        dimensions=args.dimensions,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    main()

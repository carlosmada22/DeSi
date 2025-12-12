#!/usr/bin/env python3
"""
Offline evaluation runner for DeSi.

This script:
- Loads an evaluation dataset in JSONL format (one JSON per line)
- Runs DeSi's RAGQueryEngine on each question (single-turn)
- Logs:
    - question
    - DeSi's generated answer
    - reference_answer (from dataset)
    - gold_docs (from dataset)
    - retrieved_contexts (page_content of each retrieved chunk)
    - retrieved_metadata (origin, source, etc. from each chunk's metadata)
- Writes all results to a JSONL file for later metric computation (e.g. RAGAS).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path (same pattern as main.py)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from desi.query.query import RAGQueryEngine
from desi.utils.config import DesiConfig
from desi.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_dataset(path: Path):
    """Load JSONL evaluation dataset into a list of dicts."""
    examples = []
    if not path.exists():
        logger.error(f"Dataset file not found: {path}")
        return examples

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if "question" not in ex:
                    logger.warning(
                        f"Line {line_no}: missing 'question' field, skipping."
                    )
                    continue
                examples.append(ex)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_no}: JSON decode error: {e}. Skipping.")
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run offline evaluation for DeSi.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/eval/desi_eval_dataset.jsonl",
        help="Path to JSONL file with evaluation questions.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/eval/baseline_run.jsonl",
        help="Where to write the evaluation run log (JSONL).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to .env configuration file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query.",
    )

    args = parser.parse_args()

    # Initialize logging
    setup_logging()
    logger.info("🔍 Starting DeSi offline evaluation runner")

    # Load configuration (.env)
    config = DesiConfig(args.config)
    project_root = Path(__file__).parent

    # Resolve paths used by DeSi
    db_path = str(project_root / config.db_path)
    prompt_template_path = str(project_root / "prompts" / "desi_query_prompt.md")

    logger.info(f"Using ChromaDB at: {db_path}")
    logger.info(f"Using prompt template: {prompt_template_path}")

    # Initialize RAG engine (same Ollama + Chroma setup as main.py)
    rag_engine = RAGQueryEngine(
        chroma_persist_directory=db_path,
        prompt_template_path=prompt_template_path,
        embedding_model=config.embedding_model_name,
        llm_model=config.model_name,
    )

    if not rag_engine.vector_store or not rag_engine.llm:
        logger.error(
            "Failed to initialize RAGQueryEngine (Ollama or vector store not available). "
            "Ensure Ollama is running and the vector DB has been built."
        )
        sys.exit(1)

    dataset_path = project_root / args.dataset_path
    output_path = project_root / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_dataset(dataset_path)
    if not examples:
        logger.error("No examples loaded; aborting evaluation.")
        sys.exit(1)

    logger.info(f"Writing evaluation run log to: {output_path}")

    total = len(examples)
    with output_path.open("w", encoding="utf-8") as f_out:
        for idx, ex in enumerate(examples, start=1):
            question = ex.get("question", "").strip()
            if not question:
                logger.warning(f"Example {idx} has empty question, skipping.")
                continue

            logger.info(f"[{idx}/{total}] Evaluating question: {question}")

            try:
                # Single-turn evaluation: no conversation history
                answer, docs = rag_engine.query(
                    query=question,
                    conversation_history=[],
                    top_k=args.top_k,
                )

                # Extract contexts and metadata from retrieved docs
                contexts = [d.page_content for d in docs]
                metadata = [d.metadata for d in docs]

                result_entry = {
                    "question": question,
                    "source": ex.get("source"),
                    "reference_answer": ex.get("reference_answer"),
                    "gold_docs": ex.get("gold_docs", []),
                    "desi_answer": answer,
                    "retrieved_contexts": contexts,
                    "retrieved_metadata": metadata,
                }

                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

            except KeyboardInterrupt:
                logger.warning("Interrupted by user, stopping evaluation.")
                break
            except Exception as e:
                logger.error(
                    f"Error while processing example {idx}: {e}", exc_info=True
                )

    logger.info("✅ Evaluation run complete.")


if __name__ == "__main__":
    main()

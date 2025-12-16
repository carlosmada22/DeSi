#!/usr/bin/env python3
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas.run_config import RunConfig
from tqdm import tqdm

# Prefer the modern ChatOllama
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from desi.utils.config import DesiConfig


# -----------------------------
# Logging setup
# -----------------------------
def setup_logger(log_path: Path):
    logger = logging.getLogger("ragas_eval")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Avoid duplicate handlers if re-run in same process
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def load_baseline_run(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            q = (ex.get("question") or "").strip()
            a = (ex.get("desi_answer") or "").strip()
            ctxs = ex.get("retrieved_contexts") or []
            gt = (ex.get("reference_answer") or "").strip()

            if not q or not a:
                continue

            rows.append(
                {
                    "user_input": q,
                    "response": a,
                    "retrieved_contexts": ctxs,
                    "reference": gt,
                }
            )
    return rows


def main():
    project_root = Path(__file__).parent
    _ = DesiConfig(".env")  # load env

    run_path = project_root / "data" / "eval" / "baseline_run.jsonl"
    out_csv = project_root / "data" / "eval" / "ragas_per_sample.csv"
    out_jsonl = project_root / "data" / "eval" / "ragas_per_sample.jsonl"
    log_path = project_root / "data" / "eval" / "ragas_run.log"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("Starting RAGAS evaluation with per-item progress logging")
    logger.info(f"Input: {run_path}")
    logger.info(f"Output CSV: {out_csv}")
    logger.info(f"Output JSONL: {out_jsonl}")
    logger.info(f"Log file: {log_path}")

    config = DesiConfig(".env")

    # ---------
    # Init models
    # ---------
    logger.info("Initializing Ollama judge LLM + embeddings...")
    logger.info(f"LLM model: {config.model_name}")
    logger.info(f"Embedding model: {config.embedding_model_name}")

    judge_llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.0,
        format="json",
    )

    judge_embeddings = OllamaEmbeddings(model=config.embedding_model_name)

    # Quick connectivity check: prove Ollama is actually responding
    try:
        ping = judge_llm.invoke("Reply with just the word: OK")
        logger.info(f"Ollama connectivity check OK. Model responded: {str(ping)[:200]}")
    except Exception:
        logger.error("Ollama connectivity check FAILED. Is Ollama running / reachable?")
        raise

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]

    # ---------
    # Load rows
    # ---------
    rows = load_baseline_run(run_path)
    logger.info(f"Loaded {len(rows)} evaluation rows.")

    # If you want a quick smoke test first:
    # rows = rows[:10]

    # ---------
    # Run per item with progress + incremental save
    # ---------
    all_results = []
    # Start fresh outputs
    out_jsonl.write_text("", encoding="utf-8")

    for i, row in enumerate(tqdm(rows, desc="RAGAS eval", unit="q"), start=1):
        q_preview = row["user_input"][:120].replace("\n", " ")
        logger.info(f"[{i}/{len(rows)}] START - {q_preview}")

        t0 = time.time()

        # evaluate expects a Dataset
        ds_one = Dataset.from_list([row])

        my_run_config = RunConfig(
            timeout=60,  # max seconds for a metric call
            max_workers=8,  # run 8 async calls concurrently
        )

        try:
            res = evaluate(
                dataset=ds_one,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_embeddings,
                run_config=my_run_config,
                batch_size=8,
            )
            df_one = res.to_pandas()

            # add useful debugging columns
            df_one["idx"] = i
            df_one["user_input"] = row["user_input"]
            df_one["reference"] = row["reference"]

            elapsed = time.time() - t0
            logger.info(f"[{i}/{len(rows)}] DONE in {elapsed:.1f}s")

            # Save incremental JSONL
            record = df_one.to_dict(orient="records")[0]
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            all_results.append(record)

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"[{i}/{len(rows)}] FAIL in {elapsed:.1f}s - {e}", exc_info=True
            )

            # still write a failure record so you can debug later
            fail_record = {
                "idx": i,
                "user_input": row["user_input"],
                "error": str(e),
            }
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")

    # ---------
    # Final CSV
    # ---------

    df = pd.DataFrame(all_results)
    df.to_csv(out_csv, index=False)

    logger.info(f"✅ Saved per-sample RAGAS scores to: {out_csv}")
    logger.info("📊 Overall metric means:")
    logger.info(str(df.mean(numeric_only=True)))


if __name__ == "__main__":
    main()

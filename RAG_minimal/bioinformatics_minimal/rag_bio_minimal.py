# - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
# - Generator:  google/flan-t5-small  (local CPU)
# - Input: uses ./input_data/variants.tsv + sample_metadata.tsv if available
#          else synthesizes a small cohort
#
# variants.tsv columns: sample_id, gene, impact, AF, consequence
# sample_metadata.tsv:  sample_id, cohort, sex, age
#
# pip install:
#   sentence-transformers==2.2.2 transformers==4.37.2 torch==2.2.1 pandas numpy sentencepiece

import os
import math
import random
import pathlib
import numpy as np
import pandas as pd
import torch
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_DIR = pathlib.Path("./input_data")
VARIANTS_PATH = DATA_DIR / "variants.tsv"
SAMPLES_PATH = DATA_DIR / "sample_metadata.tsv"

# -----------------------------
# Helpers: data I/O / synthesis
# -----------------------------
def ensure_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    if VARIANTS_PATH.exists() and SAMPLES_PATH.exists():
        variants = pd.read_csv(VARIANTS_PATH, sep="\t")
        samples = pd.read_csv(SAMPLES_PATH, sep="\t")
        # coerce AF
        variants["AF"] = pd.to_numeric(variants["AF"], errors="coerce").fillna(0.0)
        return variants, samples
    # synthesize small cohort
    rng = random.Random(42)
    n_samples = 60
    n_genes   = 24
    genes = [f"GENE{g:02d}" for g in range(1, n_genes + 1)]
    impacts = [
        "synonymous_variant", "missense_variant",
        "frameshift_variant", "stop_gained",
        "splice_acceptor_variant", "splice_donor_variant"
    ]
    consequences = {"synonymous_variant": "low",
                    "missense_variant": "moderate",
                    "frameshift_variant": "high",
                    "stop_gained": "high",
                    "splice_acceptor_variant": "high",
                    "splice_donor_variant": "high"}
    samples = pd.DataFrame({
        "sample_id": [f"S{idx:03d}" for idx in range(1, n_samples+1)],
        "cohort":    [rng.choice(["Discovery","Validation","Replication"]) for _ in range(n_samples)],
        "sex":       [rng.choice(["F","M"]) for _ in range(n_samples)],
        "age":       [rng.randint(20, 85) for _ in range(n_samples)],
    })
    rows = []
    for sid in samples["sample_id"]:
        # 5–20 variants per sample
        for _ in range(rng.randint(5, 20)):
            impact = rng.choice(impacts)
            af_base = {"synonymous_variant": rng.random()*0.2,
                       "missense_variant": rng.random()*0.1,
                       "frameshift_variant": rng.random()*0.01,
                       "stop_gained": rng.random()*0.02,
                       "splice_acceptor_variant": rng.random()*0.01,
                       "splice_donor_variant": rng.random()*0.01}[impact]
            rows.append({
                "sample_id": sid,
                "gene": rng.choice(genes),
                "impact": impact,
                "AF": round(af_base, 5),
                "consequence": consequences[impact],
            })
    variants = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    variants.to_csv(VARIANTS_PATH, sep="\t", index=False)
    samples.to_csv(SAMPLES_PATH, sep="\t", index=False)
    return variants, samples

def is_lof_series(impact: pd.Series) -> pd.Series:
    return impact.isin(["stop_gained","frameshift_variant","splice_acceptor_variant","splice_donor_variant"])

def af_bucket_series(af: pd.Series) -> pd.Series:
    # ultra_rare <1e-4, rare <1e-2, else common
    return np.select(
        [af < 1e-4, af < 1e-2],
        ["ultra_rare","rare"],
        default="common"
    )

# --------------------------------
# Build readable per-gene documents
# --------------------------------
def build_gene_docs(variants: pd.DataFrame) -> List[str]:
    v = variants.copy()
    v["is_lof"] = is_lof_series(v["impact"])
    v["af_bucket"] = af_bucket_series(v["AF"])

    docs: List[str] = []
    for gene, df in v.groupby("gene"):
        n_total = len(df)
        n_lof   = int(df["is_lof"].sum())
        n_rare_lof = int(((df["is_lof"]) & (df["AF"] < 0.01)).sum())
        max_af = df["AF"].max() if n_total else 0.0

        bucket_counts = df["af_bucket"].value_counts().to_dict()
        impact_counts = df["impact"].value_counts().to_dict()

        # Make a compact, text-only “fact sheet” per gene (great for retrieval):
        text = (
            f"Gene: {gene}. "
            f"Total variants: {n_total}. "
            f"Loss-of-function (LoF): {n_lof}. "
            f"Rare LoF (AF<0.01): {n_rare_lof}. "
            f"Max AF: {max_af:.5f}. "
            f"AF buckets: {bucket_counts}. "
            f"Impacts: {impact_counts}."
        )
        docs.append(text)
    return docs

# -----------------------
# Dense retrieval + LLM
# -----------------------
def embed_corpus(corpus: List[str], model: SentenceTransformer) -> torch.Tensor:
    return model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

def topk_docs(query: str, corpus: List[str], doc_embeddings: torch.Tensor, embedder: SentenceTransformer, k: int = 5) -> List[str]:
    q = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    sims = torch.matmul(q, doc_embeddings.T)  # cosine because normalized
    k = min(k, doc_embeddings.shape[0])
    idx = torch.topk(sims, k=k).indices[0].tolist()
    return [corpus[i] for i in idx]

def generate_answer(context: str, question: str, tok, model, max_new_tokens: int = 200) -> str:
    prompt = f"Use the CONTEXT to answer the QUESTION.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER (brief, factual):"
    inputs = tok(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)

# -------------
# Main routine
# -------------
def main():
    print("Loading/creating data …")
    variants, samples = ensure_dataframes()
    print(f"Variants: {variants.shape}, Samples: {samples.shape}")

    print("Building per-gene documents …")
    docs = build_gene_docs(variants)
    print(f"Docs: {len(docs)} (one per gene)")

    print("Loading embedding model …")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding documents …")
    doc_emb = embed_corpus(docs, embedder)

    print("Loading local generator (FLAN-T5) …")
    gen_model_name = "google/flan-t5-small"  # try base if you have more CPU/RAM
    tok = AutoTokenizer.from_pretrained(gen_model_name)
    gen = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    # Example bioinformatics questions you can tweak freely:
    questions = [
        "Which gene shows the most rare loss-of-function variants?",
        "Which genes have ultra_rare variants?",
        "What is the maximum allele frequency observed in GENE05?",
        "Summarize LoF burden across genes.",
    ]

    for q in questions:
        hits = topk_docs(q, docs, doc_emb, embedder, k=5)
        context = "\n".join(hits)
        ans = generate_answer(context, q, tok, gen, max_new_tokens=120)
        print("\nQ:", q)
        print("Top hit:", hits[0][:120] + ("…" if len(hits[0]) > 120 else ""))
        print("A:", ans)

if __name__ == "__main__":
    main()

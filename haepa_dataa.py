from logger import logger
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

datapath = "data/"

def load_collection(path):
    collection = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.strip().split("\t", 1)
            collection[pid] = text
    return collection


def build_small_collection(collection_path, qrels, max_negatives=50000):
    relevant_pids = set()
    for pids in qrels.values():
        relevant_pids.update(pids)

    collection = {}
    all_lines = []

    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            all_lines.append(line)

    # Add relevant passages
    for line in all_lines:
        pid, text = line.strip().split("\t", 1)
        if pid in relevant_pids:
            collection[pid] = text

    # Add random negatives
    remaining = [l for l in all_lines if l.split("\t", 1)[0] not in relevant_pids]
    sampled = random.sample(remaining, min(max_negatives, len(remaining)))

    for line in sampled:
        pid, text = line.strip().split("\t", 1)
        collection[pid] = text

    return collection

def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, pid, relevance = line.strip().split()
            if int(relevance) > 0:
                qrels.setdefault(qid, set()).add(pid)
    return qrels


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, text = line.strip().split("\t", 1)
            queries[qid] = text
    return queries


queries = load_queries(datapath + "queries.dev.tsv")
#collection = load_collection(datapath + "collection.tsv")
qrels = load_qrels(datapath + "qrels.dev.tsv")
collection = build_small_collection(
    datapath + "collection.tsv",
    qrels,
    max_negatives=20000
)


def encode(texts) -> torch.Tensor
    passage_ids = list(collection.keys())
    passage_texts = [collection[pid] for pid in passage_ids]
    passage_embeddings = encode(passage_texts)  # shape: [N, dim]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = encode(query_texts)


def compute_mrr_at_10(query_embeddings, passage_embeddings, query_ids, passage_ids, qrels):
    sim_matrix = cosine_similarity(query_embeddings, passage_embeddings)
    mrr_total = 0.0
    evaluated = 0

    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue

        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)[:10]

        reciprocal_rank = 0.0
        for rank, idx in enumerate(ranked_indices, start=1):
            pid = passage_ids[idx]
            if pid in qrels[qid]:
                reciprocal_rank = 1.0 / rank
                break

        mrr_total += reciprocal_rank
        evaluated += 1

    return mrr_total / evaluated


mrr = compute_mrr_at_10(
    query_embeddings,
    passage_embeddings,
    query_ids,
    passage_ids,
    qrels
)

print("MRR@10:", mrr)
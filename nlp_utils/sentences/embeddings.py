"""
Utility functions for quickly getting to work with sentence level
embeddings.

todo: proper documentation.
"""


import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from nlp_utils.sentences.datasets import load_sts_dataset

def cosine_sim(a, b):
    """Computes cosine similarity between a and b"""
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def compare_embeddings(embedding_one, embedding_two):
    emb_one_is_list = isinstance(embedding_one[0], (list, np.ndarray))
    emb_two_is_list = isinstance(embedding_two[0], (list, np.ndarray))

    # pairwise compare all embeddings
    if emb_one_is_list and emb_two_is_list:
        if len(embedding_one) != len(embedding_two):
            raise RuntimeError(
                "embedding one and embedding two lists not of equal length"
            )
        return np.array([
            cosine_sim(e1, e2)
            for e1, e2 in zip(embedding_one, embedding_two)
        ])

    # embedding one compared against all of embedding two
    elif emb_two_is_list:
        return [cosine_sim(embedding_one, e2) for e2 in embedding_two]
    # compare embedding one against embedding two
    elif not emb_one_is_list:
        return cosine_sim(embedding_one, embedding_two)

    # embedding one is a list of embeddings but embedding two is one embedding
    raise RuntimeError(
        "Either Embedding one and two are lists of embeddings, else only embedding two."
    )


def rescale_numeric(cos_dist, cur_min=-1, cur_max=1, new_min=0, new_max=5):
    # percent of measurement on current scale
    cur_perc = (cos_dist - cur_min) / (cur_max - cur_min)

    # for scaling the measurement to the new range
    scaling_fct = (new_max - new_min) + new_min
    return cur_perc * scaling_fct


def get_embedding_distances(s1_embeddings, s2_embeddings, scale=True):
    distances = compare_embeddings(s1_embeddings, s2_embeddings)

    # return cosine similarity unscaled
    if not scale:
        return distances

    return rescale_numeric(distances)


def eval_embedding_correlation(distances, truth_vector):
    pearson_corr, pearson_pval = pearsonr(distances, truth_vector)
    spearman_corr, spearman_pval = spearmanr(distances, truth_vector)

    return pd.DataFrame({
        "type": ["pearson", "spearman"],
        "corr": [pearson_corr, spearman_corr],
        "pval": [pearson_pval, spearman_pval],
    })


def benchmark_model(sts_df, encode_func):
    s1_embeddings = encode_func(sts_df.sent1.values)
    s2_embeddings = encode_func(sts_df.sent2.values)
    gold_standard = sts_df.score.values

    embedding_distances = get_embedding_distances(
        s1_embeddings,
        s2_embeddings)

    return(eval_embedding_correlation(embedding_distances, gold_standard))


def evaluate_sts_model(model,
                       model_name,
                       dataset=None,
                       evaluator_name="sts-test",
                       outdir="../data/inference/sts"):
    if dataset is None:
        _, _, dataset = load_sts_dataset()

    sts_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dataset,
        name=evaluator_name
    )

    outdir = f"{outdir}/{model_name}"
    os.makedirs(outdir, exist_ok=True)

    results_path = \
        f"{outdir}/similarity_evaluation_{evaluator_name}_results.csv"

    if os.path.exists(results_path):
        os.remove(results_path)

    print("evaluating model...")
    sts_evaluator(model, output_path=outdir)

    return pd.read_csv(results_path)


def run_evaluator(model,
                  model_name,
                  dataset=None,
                  evaluator_name="sts-test",
                  outdir="../data/inference/sts"):
    if dataset is None:
        _, _, dataset = load_sts_dataset()

    sts_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dataset,
        name=evaluator_name
    )

    outdir = f"{outdir}/{model_name}"
    os.makedirs(outdir, exist_ok=True)

    results_path = \
        f"{outdir}/similarity_evaluation_{evaluator_name}_results.csv"

    if os.path.exists(results_path):
        os.remove(results_path)

    print("evaluating model...")
    sts_evaluator(model, output_path=outdir)

    return pd.read_csv(results_path)


def label_sts_dataset(dataset, encode_func, group_name=None):
    print("labelling dataset...")
    dataset["embedding_1"] = dataset.sentence_1.apply(encode_func)
    dataset["embedding_2"] = dataset.sentence_2.apply(encode_func)

    embedding_dists = get_embedding_distances(
        dataset["embedding_1"],
        dataset["embedding_2"],
        scale=False
    )
    dataset["distance"] = rescale_numeric(
        embedding_dists,
        cur_min=min(embedding_dists),
        new_max=1)

    dataset["residual"] = dataset["score"] - dataset["distance"]
    dataset["abs_residual"] = dataset.residual.apply(abs)
    dataset["sq_residual"] = dataset.residual.apply(lambda x: pow(x, 2))

    if group_name is not None:
        dataset["group"] = group_name

    return dataset

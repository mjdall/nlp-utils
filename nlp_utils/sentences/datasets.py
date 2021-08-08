import os
import gzip
import csv
import pandas as pd


from sentence_transformers import util
from sentence_transformers.readers import *



def cache_sts_eval_data(
    dataset = "stsbenchmark",
    save_dir = "../data/inference",
    force = False):
    file_path = f"{save_dir}/{dataset}.tsv.gz"

    if not os.path.exists(file_path) or force:
        print("fetching dataset...")
        fetch_path = f"https://sbert.net/datasets/{dataset}.tsv.gz"
        util.http_get(fetch_path, file_path)
    else:
        print("dataset already exsits...")
    
    return file_path


def load_sts_dataset(dataset_gz=None, as_pd=False):
    if dataset_gz is None:
        dataset_gz = cache_sts_eval_data()

    test_sts_samples = []
    dev_sts_samples = []
    train_sts_samples = []
    with gzip.open(dataset_gz, "rt", encoding="utf8") as file_in:
        reader = csv.DictReader(
            file_in,
            delimiter="\t",
            quoting=csv.QUOTE_NONE)

        for idx, row in enumerate(reader):
            score = row.get("label",)

            if "score" in row:
                # rescale to 0 - 1
                score = float(row.get("score", None)) / 5.0

            assert score is not None, "could not detect label"

            if as_pd:
                inp_example = {
                    "sentence_1": row["sentence1"],
                    "sentence_2": row["sentence2"],
                    "score": score,
                }
            else:
                inp_example = InputExample(
                    texts=[
                        row["sentence1"],
                        row["sentence2"]
                    ],
                    label=score)


            if row["split"] == "test":
                test_sts_samples.append(inp_example)
            elif row["split"] == "dev":
                dev_sts_samples.append(inp_example)
            else:
                train_sts_samples.append(inp_example)

    if as_pd:
        train_sts_samples = pd.DataFrame(train_sts_samples)
        dev_sts_samples = pd.DataFrame(dev_sts_samples)
        test_sts_samples = pd.DataFrame(test_sts_samples)

    return train_sts_samples, dev_sts_samples, test_sts_samples

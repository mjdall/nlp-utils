# nlp-utils
NLP utility functions I find myself using often.
Mostly sentence transformer based. Embedding functions can be generalised for any embeddings derived via metric learning.

## Modules
### Embeddings
Function for computing distances, running pearsons and spearmans corr tests, etc.

Import with: `from nlp_utils.sentences.embeddings import ...`.

### Visualisation
Functions for visualising embeddings, main two to use are:
* `run_umap(df, embedding_col, n_components=2)` will label the input df with a components column.
* use `vis_components` to then visualise umap components with plotly.
    * if `zcol` is provided, a 3d plot will be returned instead.

Import with:
`from nlp_utils.sentences.visualisation import run_umap, vis_components`

### Datasets
Functions for downloading and loading datasets.
* `load_sts_dataset` will load an sts dataset, if not cached already, it will call the caching function: `cache_sts_eval_data`.

Import with:
`from nlp_utils.sentences.datasets import ...`


## Example Notebook
See `examples/embedding_compare.ipynb` for example usage.

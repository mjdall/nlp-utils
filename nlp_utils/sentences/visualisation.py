import umap
import plotly.express as px
import pandas as pd


def run_umap(df, embedding_col="embedding", n_components=2, **umap_kwargs):
    """
    Runs umap on an array of vectors
        and returns a dataframe with the embeddings and class labels.
    """
    umap_model = umap.UMAP(n_components=3, **umap_kwargs)
    embeddings = [list(e) for e in df[embedding_col].values]
    umap_comps = umap_model.fit_transform(embeddings)

    for n in range(0, n_components):
        comp_no = n + 1
        df[f"component_{comp_no}"] = umap_comps[:, n]

    return df


def vis_components(
    components_df,
    xcol="component_1",
    ycol="component_2",
    zcol=None,
    colour_col="class",
    hover_col="sentence",
    height=780,
    width=1366):
    """
    Plots a component dataframe in an interactive 2d or 3d plotly plot.
    Returns the plotly figure.
    """
    if zcol is None:
        fig = px.scatter(
            components_df,
            x=xcol,
            y=ycol,
            color=colour_col,
            hover_name=hover_col)
    else:
        fig = px.scatter_3d(
            components_df,
            x=xcol,
            y=ycol,
            z=zcol,
            color=colour_col,
            hover_name=hover_col)

    fig.update_traces(
        marker=dict(size=5),
        selector=dict(mode="markers"))

    fig.update_layout(
        margin=dict(l=16, r=16, t=16, b=16),
        height=height, width=width
    )

    return(fig)


def scatter_compare_embeddings(labelled_df,
                              dist_col="distance",
                              truth_col="score",
                              colour_col="group"):
    fig = px.scatter(
        labelled_df,
        x=truth_col,
        y=dist_col,
        color=colour_col)

    fig.update_layout(
        margin=dict(l=16, r=16, t=16, b=16),
        height=768, width=1366
    )

    return fig

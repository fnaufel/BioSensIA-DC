import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import biosensia_target_fishing as bsia
    import polars as pl
    import altair as alt

    alt.data_transformers.enable("vegafusion")
    return alt, bsia


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pocket candidates

    We built an LMDB file from the `external/DrugCLIP/data/pdb/combine_set` directory, using function `build_candidate_pockets_frame` in file [`biosensia_target_fishing.py`](biosensia_target_fishing.py). Here is a summary:
    """)
    return


@app.cell
def _(bsia):
    df = bsia.build_candidate_pockets_frame(lmdb_path='data/candidate_pockets.lmdb')
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(alt, df):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "pocket_atoms:Q",
                bin=alt.Bin(maxbins=40),
                title="Number of pocket atoms",
            ),
            y=alt.Y("count():Q", title="Number of pockets"),
            tooltip=[alt.Tooltip("count():Q", title="Number of pockets")],
        )
        .properties(
            title="Distribution of pocket atom counts",
            width="container",
            height=360,
        )
    )
    chart
    return


if __name__ == "__main__":
    app.run()

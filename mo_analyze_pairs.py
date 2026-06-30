import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    df = pl.read_parquet(
        "data/biosensia_finetune/training_data_pairs.parquet"
    ).with_columns(
        pl.col('lmdb_key').cast(pl.Int32)
    )

    df.head()
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LMDB keys
    """)
    return


@app.cell
def _(df, pl):
    df.select(
        pl.col('lmdb_key').min().alias('min'),
        pl.col('lmdb_key').max().alias('max')
    )
    return


@app.cell
def _(df, pl):
    df.group_by('split').agg(
        pl.len()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uniprot accessions
    """)
    return


@app.cell
def _(df):
    df.n_unique(
        subset=['pocket_uniprot_accessions']
    )
    return


if __name__ == "__main__":
    app.run()

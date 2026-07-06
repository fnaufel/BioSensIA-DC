import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    from biosensia_target_fishing_benchmark import write_positive_pairs_from_lmdb

    return pl, write_positive_pairs_from_lmdb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate positives dataframe and save to parquet file
    """)
    return


@app.cell
def _(write_positive_pairs_from_lmdb):
    write_positive_pairs_from_lmdb(
        "data/biosensia_finetune/valid.lmdb",
        "runs/target_fishing_benchmarks/valid_positives.parquet",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read and examine positives data frame
    """)
    return


@app.cell
def _(pl):
    df_pos = pl.read_parquet('./runs/target_fishing_benchmarks/valid_positives.parquet')
    return (df_pos,)


@app.cell
def _(df_pos):
    df_pos
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read and check molecule file
    """)
    return


@app.cell
def _(pl):
    from lmdb_helpers import read_lmdb_records

    _valid_records = read_lmdb_records("data/biosensia_finetune/valid.lmdb")
    df_valid_lmdb = pl.DataFrame(_valid_records)
    df_valid_lmdb
    return (df_valid_lmdb,)


@app.cell
def _(df_valid_lmdb):
    df_valid_lmdb.glimpse()
    return


if __name__ == "__main__":
    app.run()

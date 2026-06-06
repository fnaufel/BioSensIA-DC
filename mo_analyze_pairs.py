import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from mo_shell import sh 

    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    alt.data_transformers.enable("vegafusion")
    return


@app.cell
def _():
    import scratch.analyze_pdb_complexes as apc

    df = apc.build_complexes_frame(limit=20)
    df
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import subprocess as sp
    from pathlib import Path


    def fenced_block(text: str, language: str = "bash") -> str:
        """
        Return a Markdown fenced code block, choosing a fence long enough
        not to conflict with the block contents.
        """
        fence = "```"
        while fence in text:
            fence += "`"

        return f"{fence}{language}\n{text.rstrip()}\n{fence}"


    def sh(
        command: str,
        cwd: str | Path = ".",
        language: str = "bash",
    ) -> mo.Html:
        result = sp.run(
            ["bash", "-lc", command],
            cwd=Path(cwd),
            capture_output=True,
            text=True,
            check=False,
        )

        parts = []

        if result.stdout:
            parts.append("**stdout**")
            parts.append(fenced_block(result.stdout, language))

        if result.stderr:
            parts.append("**stderr**")
            parts.append(fenced_block(result.stderr, language))

        if result.returncode != 0:
            parts.append(f"**exit status:** `{result.returncode}`")

        if not parts:
            parts.append("Command produced no output.")

        return mo.md("\n\n".join(parts))

    return Path, sh


@app.cell
def _():
    import importlib
    import polars as pl
    import altair as alt
    import lmdb_helpers

    alt.data_transformers.enable("vegafusion")
    return alt, importlib, lmdb_helpers


@app.cell
def _(importlib):
    import biosensia_target_fishing as tf
    importlib.reload(tf)
    return (tf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pocket candidates

    ## Building the LMDB file of pockets

    We built an LMDB file from the `external/DrugCLIP/data/pdb/combine_set` directory, using function `build_candidate_pockets_frame` in file [`biosensia_target_fishing.py`](biosensia_target_fishing.py). Here is a summary:
    """)
    return


@app.cell
def _(tf):
    df = tf.build_candidate_pockets_frame(lmdb_path='data/candidate_pockets.lmdb')
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


@app.cell
def _(mo):
    mo.md(r"""
    This is the set of candidate pockets that will be encoded and ranked according to the similarity to the query molecule(s).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting `candidate_pockets.lmdb`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each record consists of

    - A numeric ASCII string as the key.
    - As the value, a pickled dictionary with keys:
      - 'pocket': the PDB 4-charcater code of the complex.
      - 'pocket_atoms': the list of atoms comprising the pocket.
      - 'pocket_coordinates': the list of 3D coordinates of the atoms.
    """)
    return


@app.cell
def _(lmdb_helpers):
    lmdb_helpers.read_lmdb_records('data/candidate_pockets.lmdb', head_n=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The query molecule(s)

    ## Indexing the known molecules

    The ligand molecule will be passed as a query to the target fishing function as an LMDB file. To save this LMDB query file, we will search the LMDB file DrugCLIP uses to store data about all known molecules --- `external/DrugCLIP/mols.lmdb`.

    However, this file (like `candidate_pockets.lmdb`, described above) has sequential numerical values as keys, so all searches would have to be sequential, taking too long, as the file contains 2,942,719 records.

    To avoid this, the function `build_mol_lmdb_index` creates another LMDB file to index the molecules by canonical SMILES formula.


    ## Building the query molecule(s) `lmdb` file

    The function `create_mol_lmdb` will receive the specification of the molecule(s), search the index, get the data from the local source (default `external/DrugCLIP/mols.lmdb`) and save an LMDB file. The molecule(s) of interest can be given as follows:

    - SMILES strings are supported directly.
    - DrugCLIP IDs are matched against the local source LMDB.

    Molecules missing from the local source are resolved through PubChem when `download_missing` is true; use `cid:2244` or `2244` for a PubChem CID, or a PubChem compound name.

    Let's build a query for okadaic acid. We are using the SMILES formula from <https://pubchem.ncbi.nlm.nih.gov/compound/Okadaic-Acid>:
    """)
    return


@app.cell
def _(tf):
    tf.create_mol_lmdb(
        r'C[C@@H]1CC[C@]2(CCCCO2)O[C@@H]1[C@@H](C)C[C@@H]([C@@H]3C(=C)[C@H]([C@H]4[C@H](O3)CC[C@]5(O4)CC[C@@H](O5)/C=C/[C@@H](C)[C@@H]6CC(=C[C@@]7(O6)[C@@H](CC[C@H](O7)C[C@](C)(C(=O)O)O)O)C)O)O',
        'data/query_mol.lmdb'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Surprisingly, okadaic acid is not in the local data.

    Let's check the LMDB file:
    """)
    return


@app.cell
def _(lmdb_helpers):
    lmdb_helpers.read_lmdb_records('data/query_mol.lmdb')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Running the query

    Change to the correct directory and run the bash script:

    ```bash
    cd external/DrugCLIP/
    ./target_fishing.sh
    ```

    The first time the script is run, embeddings for all candidate pockets will be computed and saved to `external/DrugCLIP/data/pocket_emb/pockets_candidate_pockets.lmdb.pkl`. This will take a few minutes.

    The results will be saved in `external/DrugCLIP/data/pocket_emb/ranked_pockets.txt`.

    Here are the first 50 pockets:
    """)
    return


@app.cell
def _(sh):
    sh(
        'cat -n external/DrugCLIP/data/pocket_emb/ranked_pockets.txt | head -n50',
        cwd='..',
        language='text'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As a polars dataframe:
    """)
    return


@app.cell
def _(tf):
    df_ranked = tf.build_ranked_pockets_frame('../external/DrugCLIP/data/pocket_emb/ranked_pockets.txt')
    df_ranked
    return (df_ranked,)


@app.cell
def _(Path, df_ranked):
    import pyarrow

    html = df_ranked.to_pandas().to_html(
        index=False,
        render_links=True,
        escape=False,
    )

    Path("../external/DrugCLIP/data/pocket_emb/ranked_pockets.html").write_text(
        f"""<!doctype html>
    <html>
    <head><meta charset="utf-8"><title>Ranked pockets</title></head>
    <body>
    {html}
    </body>
    </html>
    """,
        encoding="utf-8",
    )
    return


if __name__ == "__main__":
    app.run()

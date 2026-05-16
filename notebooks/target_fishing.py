import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import importlib
    import lmdb
    import pickle
    import polars as pl
    import altair as alt

    alt.data_transformers.enable("vegafusion")
    return alt, importlib, lmdb, pickle


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
def _(lmdb, pickle):
    def read_lmdb(lmdb_path, head_n=None):

        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )  

        txn = env.begin()
        # LMDB cursor order is lexicographic byte order, not numeric order,
        # so we sort the keys:
        keys = sorted(
            txn.cursor().iternext(values=False), 
            key=lambda k: int(k.decode('ascii'))
        )
    
        out_dict = {}
        i = 1

        for idx in keys:
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            out_dict[idx] = data
            i += 1
            if head_n is not None and i > head_n:
                break
            keys = txn.cursor()

        env.close()
        return out_dict


    return (read_lmdb,)


@app.cell
def _(read_lmdb):
    read_lmdb('data/candidate_pockets.lmdb', 5)
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

    Let's build a query for okadaic acid:
    """)
    return


@app.cell
def _(tf):
    tf.create_mol_lmdb(
        r'C=C1[C@@H](O)[C@@H]2O[C@]3(CC[C@H](/C=C/[C@@H](C)[C@@H]4CC(C)=C[C@@]5(O[C@H](C[C@@](C)(O)C(=O)O)CC[C@H]5O)O4)O3)CC[C@H]2O[C@@H]1[C@@H](O)C[C@H](C)[C@H]1O[C@@]2(CCCCO2)CC[C@H]1C',
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
def _(read_lmdb):
    read_lmdb('data/query_mol.lmdb')
    return


if __name__ == "__main__":
    app.run()

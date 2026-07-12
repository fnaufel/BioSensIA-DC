import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # *Benchmark* do BioSensIA: primeira tentativa
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introdução

    Em junho e julho de 2026, os ajustes descritos do relatório "BioSensIA-DC: *fine-tuning*" foram implementados.

    O passo seguinte foi realizar um primeiro *benchmark*. Para tanto, foi preciso definir:

    - Um **conjunto de validação** contendo as moléculas a ser submetidas como consulta.
    - Uma biblioteca de ***pockets* candidatos**.
    - Uma **tabela de positivos** contendo a associação entre as moléculas do conjunto de validação e seus *pockets* positivos.
    - **Métricas** a serem avaliadas.

    Este documento descreve as decisões tomadas, os problemas encontrados, as hipóteses formuladas e as soluções propostas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conjunto de validação

    Os dados em `data/biosensia_finetune/valid.lmdb` foram reservados para validação. Este foi o arquivo submetido como consulta no *benchmark*.

    Este conjunto de dados foi construído a partir do conteúdo do arquivo `train_no_test_af.zip`, que estava no diretório do Google Drive indicado no [README do DrugCLIP](external/DrugCLIP/README.md).

    O arquivo original continha, para cada complexo, os campos

    - `atoms`: lista de átomos do ligante.
    - `coordinates`: lista de conformações geradas pelo RDKit para o ligante. Cada conformação é uma matriz de dimensões $\text{número de átomos} \times 3$.
    - `pocket_atoms`: lista de átomos do *pocket*.
    - `pocket_coordinates`: uma **única** geometria do *pocket*.
    - `mol`: objeto RDKit para o ligante.
    - `smi`: fórmula SMILES do ligante.
    - `pocket`: identificador PDB do *pocket*.

    Nós acrescentamos os seguintes campos, basicamente contendo metadados sobre os ligantes e os *pockets*:

    - `ligand_key`.
    - `pocket_key`.
    - `biosensia_ligand_policy`.
    - `biosensia_pocket_policy`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conteúdo
    """)
    return


@app.cell
def _(load_lmdb_or_read_parquet_file):
    df_valid = load_lmdb_or_read_parquet_file('data/biosensia_finetune/valid.lmdb')
    return (df_valid,)


@app.cell
def _(df_valid):
    df_valid.glimpse()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Não há repetição de *pockets* nem de ligantes no conjunto de validação:
    """)
    return


@app.cell
def _(df_valid, pl):
    df_valid.select(
        pl.col('smi').n_unique(),
        pl.col('pocket').n_unique(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Biblioteca de *pockets* candidatos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para poupar trabalho, usamos, como conjunto de *pockets* candidatos, o mesmo arquivo que foi usado para fazer o *fine-tuning* do BioSensIA-DC: `data/biosensia_finetune/train.lmdb`.

    Este conjunto de dados foi construído a partir do conteúdo do arquivo `train_no_test_af.zip`, que estava no diretório do Google Drive indicado no [README do DrugCLIP](external/DrugCLIP/README.md).

    Segundo o README, o arquivo original foi compilado a partir do [PDBbind](https://www.pdbbind-plus.org.cn), **com registros adicionais criados por HomoAug**, a estratégia de aumento de dados proposta pelos autores do DrugCLIP.

    O arquivo original continha, para cada complexo, os campos

    - `atoms`: lista de átomos do ligante.
    - `coordinates`: lista de conformações geradas pelo RDKit para o ligante. Cada conformação é uma matriz de dimensões $\text{número de átomos} \times 3$.
    - `pocket_atoms`: lista de átomos do *pocket*.
    - `pocket_coordinates`: uma **única** geometria do *pocket*.
    - `mol`: objeto RDKit para o ligante.
    - `smi`: fórmula SMILES do ligante.
    - `pocket`: identificador PDB do *pocket*.

    Nós acrescentamos os seguintes campos, basicamente contendo metadados sobre os ligantes e os *pockets*:

    - `ligand_key`.
    - `pocket_key`.
    - `biosensia_ligand_policy`.
    - `biosensia_pocket_policy`.
    """)
    return


@app.cell
def _(load_lmdb_or_read_parquet_file):
    df_candidates = load_lmdb_or_read_parquet_file('data/biosensia_finetune/train.lmdb')
    return (df_candidates,)


@app.cell
def _(df_candidates):
    df_candidates.glimpse()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No conjunto de *pockets* candidatos, não existe par que apareça só uma vez!
    """)
    return


@app.cell
def _(df_candidates, mo, pl):
    df_unique_pairs = df_candidates.group_by(
        'pocket', 'smi'
    ).len().sort('len', descending=True)

    mo.md(
        f'''
        - Número de registros no arquivo: {df_candidates.height}.
        - Número de pares únicos: {df_unique_pairs.height}.
        - Número mínimo de repetições: {df_unique_pairs.select(pl.col('len').min()).item()}.
        - Número máximo de repetições: {df_unique_pairs.select(pl.col('len').max()).item()}.
        '''
    )
    return (df_unique_pairs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    É importante verificar os identificadores dos *pockets*:
    """)
    return


@app.cell
def _(df_unique_pairs, pl):
    df_candidate_pocket_names = df_unique_pairs.select(
        'pocket',
        id_length=pl.col('pocket').str.len_chars(),
    )

    df_candidate_pocket_names.select(
        pl.col('id_length').unique()
    )
    return (df_candidate_pocket_names,)


@app.cell
def _(df_candidate_pocket_names, pl, re):
    PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
    all_pocket_ids_match = df_candidate_pocket_names.select(
        pl.col('pocket').str.contains(PDB_ID_RE.pattern).all()
    ).item()
    all_pocket_ids_match
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusões

    - Todos os ids de *pockets* no conjunto de *pockets* candidatos são da forma usada pelo PDB (4 caracteres alfanuméricos, dos quais o primeiro é um dígito).
    - Todos os pares *pocket*-ligante aparecem repetidos 2, 3, ou 4 vezes.
    - Esta repetição deve ter sido causada pela estratégia de aumento de dados (HomoAug).
    - Se a HomoAug gerou pares novos, os *pockets* não ganharam ids novos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Helpers and imports
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import lmdb_helpers as lm
    from pathlib import Path
    from pprint import pprint
    import polars as pl
    import re

    return Path, lm, pl, re


@app.cell
def _(Path, lm, pl):
    def load_lmdb_or_read_parquet_file(path):
        parquet_path = Path("scratch") / f"{Path(path).stem}.parquet"
        if parquet_path.exists():
            print(f'Reading PARQUET file {parquet_path}.')
            df = pl.read_parquet(parquet_path)
        else:
            print(f'Reading LMDB file {path}.')
            records = lm.read_lmdb_records(path)
            # Represent objects as strings to write to parquet:
            df = pl.DataFrame(records).with_columns(
                pl.col('mol').map_elements(
                    lambda value: None if value is None else str(value),
                    return_dtype=pl.String,
                )
            )
            print(f'Writing PARQUET file {parquet_path}.')
            df.write_parquet(str(parquet_path))
        return df

    return (load_lmdb_or_read_parquet_file,)


if __name__ == "__main__":
    app.run()

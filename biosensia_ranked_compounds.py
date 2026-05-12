"""Build enriched DrugCLIP retrieval result tables without editing DrugCLIP.

This module mirrors ``external/DrugCLIP/unimol/retrieval.py::main`` up to the
``task.retrieve_mols`` call, then converts the returned SMILES and scores into
a Polars DataFrame with optional PubChem and ChEMBL enrichment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pprint
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import lmdb
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)

PUBCHEM_PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "BioSensIA-DC/0.1 (compound retrieval enrichment)"


def build_drugclip_args(
    *,
    drugclip_dir: str | Path = "external/DrugCLIP",
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    mol_path: str | Path | None = None,
    pocket_path: str | Path = "data/2ie4.lmdb",
    emb_dir: str | Path | None = None,
    results_path: str | Path | None = None,
    top_k: int = 10000,
    batch_size: int = 8,
    batch_size_valid: int = 8,
    num_workers: int = 8,
    seed: int = 1,
    fp16: bool = True,
    cpu: bool = False,
):
    """Create the Uni-Core args object used by DrugCLIP retrieval.

    The defaults correspond to ``external/DrugCLIP/retrieval.sh``, but paths are
    resolved from the repository root so this wrapper can run outside the
    DrugCLIP directory.
    """

    from unicore import options, utils

    drugclip_dir = Path(drugclip_dir).resolve()
    user_dir = drugclip_dir / "unimol"
    data_dir = Path(data_dir).resolve() if data_dir else drugclip_dir / "data"
    checkpoint_path = (
        Path(checkpoint_path).resolve()
        if checkpoint_path
        else drugclip_dir / "checkpoint_best.pt"
    )
    mol_path = Path(mol_path).resolve() if mol_path else drugclip_dir / "mols.lmdb"
    pocket_path = Path(pocket_path).resolve()
    emb_dir = Path(emb_dir).resolve() if emb_dir else drugclip_dir / "data" / "emb"
    results_path = (
        Path(results_path).resolve() if results_path else drugclip_dir / "test"
    )

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    # Import DrugCLIP's custom task/model registrations before parsing args.
    utils.import_user_module(argparse.Namespace(user_dir=str(user_dir)))

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument(
        "--pocket-path", type=str, default="", help="path for pocket data"
    )
    parser.add_argument(
        "--emb-dir", type=str, default="", help="path for saved embedding data"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help="number of top-ranked molecules to write",
    )
    options.add_model_args(parser)

    input_args = [
        str(data_dir),
        "--arch",
        "drugclip",
        "--batch-size",
        str(batch_size),
        "--batch-size-valid",
        str(batch_size_valid),
        "--ddp-backend",
        "c10d",
        "--emb-dir",
        str(emb_dir),
        "--fp16-init-scale",
        "4",
        "--fp16-scale-window",
        "256",
        "--log-format",
        "simple",
        "--log-interval",
        "100",
        "--loss",
        "in_batch_softmax",
        "--max-pocket-atoms",
        "256",
        "--mol-path",
        str(mol_path),
        "--num-workers",
        str(num_workers),
        "--path",
        str(checkpoint_path),
        "--pocket-path",
        str(pocket_path),
        "--results-path",
        str(results_path),
        "--seed",
        str(seed),
        "--task",
        "drugclip",
        "--top-k",
        str(top_k),
        "--user-dir",
        str(user_dir),
        "--valid-subset",
        "test",
    ]
    if fp16:
        input_args.append("--fp16")
    if cpu:
        input_args.append("--cpu")

    return options.parse_args_and_arch(parser, input_args=input_args)


def retrieve_mols_from_drugclip(args) -> tuple[list[str], np.ndarray]:
    """Load DrugCLIP and return ``task.retrieve_mols`` names and scores."""

    from unicore import checkpoint_utils, tasks

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    LOGGER.info("loading model(s) from %s", args.path)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    model.eval()
    LOGGER.info(
        "DrugCLIP retrieval args:\n%s",
        pprint.pformat(vars(args), sort_dicts=True),
    )
    return task.retrieve_mols(
        model,
        args.mol_path,
        args.pocket_path,
        args.emb_dir,
        args.top_k,
    )


def build_ranked_compounds_frame(
    smiles: Iterable[str],
    scores: Iterable[float],
    *,
    mol_lmdb_path: str | Path | None = None,
    enrich_pubchem: bool = False,
    pubchem_cache_path: str | Path | None = "data/pubchem_compounds.jsonl",
    pubchem_limit: int | None = None,
    enrich_chembl: bool = False,
    chembl_cache_path: str | Path | None = "data/chembl_compounds.jsonl",
    chembl_limit: int | None = None,
):
    """Build a Polars DataFrame from DrugCLIP retrieval output."""

    pl = _require_polars()
    rows = []
    for rank, (smi, score) in enumerate(zip(smiles, scores), start=1):
        rows.append(
            {
                "rank": rank,
                "drugclip_score": float(score),
                "smiles": smi,
                **_rdkit_descriptors(smi),
            }
        )

    df = pl.DataFrame(rows)

    if mol_lmdb_path is not None:
        lmdb_rows = collect_lmdb_metadata_by_smiles(mol_lmdb_path, df["smiles"])
        df = df.join(pl.DataFrame(lmdb_rows), on="smiles", how="left")

    if enrich_pubchem:
        pubchem_rows = enrich_pubchem_by_inchikey(
            df["inchikey"].drop_nulls().unique().to_list(),
            cache_path=pubchem_cache_path,
            limit=pubchem_limit,
        )
        df = df.join(pl.DataFrame(pubchem_rows), on="inchikey", how="left")

    if enrich_chembl:
        chembl_rows = enrich_chembl_by_inchikey(
            df["inchikey"].drop_nulls().unique().to_list(),
            cache_path=chembl_cache_path,
            limit=chembl_limit,
        )
        df = df.join(pl.DataFrame(chembl_rows), on="inchikey", how="left")

    return df


def collect_lmdb_metadata_by_smiles(
    mol_lmdb_path: str | Path, wanted_smiles: Iterable[str]
) -> list[dict[str, Any]]:
    """Aggregate DrugCLIP ``IDs`` and ``subset`` values by SMILES.

    ``task.retrieve_mols`` returns SMILES, not LMDB row indices. If a SMILES
    appears multiple times in the LMDB, exact row-level IDs cannot be recovered
    from those return values alone, so this function returns all matching IDs.
    """

    wanted = {smi for smi in wanted_smiles if smi is not None}
    by_smiles: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"drugclip_ids": set(), "drugclip_subsets": set()}
    )

    env = lmdb.open(
        str(mol_lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin() as transaction:
            for _, value in transaction.cursor():
                record = _loads_lmdb_record(value)
                smi = record.get("smi")
                if smi not in wanted:
                    continue
                if record.get("IDs"):
                    by_smiles[smi]["drugclip_ids"].add(record["IDs"])
                if record.get("subset"):
                    by_smiles[smi]["drugclip_subsets"].add(record["subset"])
    finally:
        env.close()

    rows = []
    for smi in sorted(wanted):
        ids = sorted(by_smiles[smi]["drugclip_ids"])
        subsets = sorted(by_smiles[smi]["drugclip_subsets"])
        rows.append(
            {
                "smiles": smi,
                "drugclip_lmdb_match_count": len(ids),
                "drugclip_ids": ids,
                "drugclip_subsets": subsets,
            }
        )
    return rows


def enrich_pubchem_by_inchikey(
    inchikeys: Iterable[str],
    *,
    cache_path: str | Path | None = "data/pubchem_compounds.jsonl",
    limit: int | None = None,
    delay_seconds: float = 0.25,
    timeout_seconds: float = 20,
) -> list[dict[str, Any]]:
    """Fetch PubChem metadata for InChIKeys, using a JSONL cache."""

    cache = _read_jsonl_cache(cache_path, "inchikey") if cache_path else {}
    rows = []
    requested = [key for key in inchikeys if key]
    if limit is not None:
        requested = requested[:limit]

    requests = [
        {
            "index": index,
            "inchikey": inchikey,
            "cached": inchikey in cache,
            "url": "" if inchikey in cache else _pubchem_property_url(inchikey),
        }
        for index, inchikey in enumerate(requested, start=1)
    ]

    for request in _iter_enrichment_requests("PubChem", requests):
        inchikey = request["inchikey"]
        if request["cached"]:
            rows.append(cache[inchikey])
            continue

        url = request["url"]
        row = _fetch_pubchem_one(
            inchikey,
            url=url,
            timeout_seconds=timeout_seconds,
        )
        rows.append(row)
        if cache_path:
            _append_jsonl(cache_path, row)
        if request["index"] < len(requests):
            time.sleep(delay_seconds)

    return rows


def enrich_chembl_by_inchikey(
    inchikeys: Iterable[str],
    *,
    cache_path: str | Path | None = "data/chembl_compounds.jsonl",
    limit: int | None = None,
    delay_seconds: float = 0.25,
    timeout_seconds: float = 20,
) -> list[dict[str, Any]]:
    """Fetch ChEMBL molecule IDs for InChIKeys, using a JSONL cache."""

    cache = _read_jsonl_cache(cache_path, "inchikey") if cache_path else {}
    rows = []
    requested = [key for key in inchikeys if key]
    if limit is not None:
        requested = requested[:limit]

    requests = [
        {
            "index": index,
            "inchikey": inchikey,
            "cached": inchikey in cache,
            "url": "" if inchikey in cache else _chembl_molecule_url(inchikey),
        }
        for index, inchikey in enumerate(requested, start=1)
    ]

    for request in _iter_enrichment_requests("ChEMBL", requests):
        inchikey = request["inchikey"]
        if request["cached"]:
            rows.append(cache[inchikey])
            continue

        url = request["url"]
        row = _fetch_chembl_one(
            inchikey,
            url=url,
            timeout_seconds=timeout_seconds,
        )
        rows.append(row)
        if cache_path:
            _append_jsonl(cache_path, row)
        if request["index"] < len(requests):
            time.sleep(delay_seconds)

    return rows


def _fetch_pubchem_one(
    inchikey: str, *, url: str | None = None, timeout_seconds: float = 20
) -> dict[str, Any]:
    url = url or _pubchem_property_url(inchikey)
    try:
        data = _get_json(url, timeout_seconds=timeout_seconds)
        props = data["PropertyTable"]["Properties"][0]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "inchikey": inchikey,
                "pubchem_status": "not_found",
                "pubchem_cid": None,
                "pubchem_name": None,
                "pubchem_url": None,
            }
        raise

    cid = props.get("CID")
    return {
        "inchikey": inchikey,
        "pubchem_status": "ok",
        "pubchem_cid": cid,
        "pubchem_name": props.get("Title") or props.get("IUPACName"),
        "pubchem_iupac_name": props.get("IUPACName"),
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
        if cid is not None
        else None,
        "pubchem_smiles": props.get("SMILES"),
        "pubchem_connectivity_smiles": props.get("ConnectivitySMILES"),
        "pubchem_formula": props.get("MolecularFormula"),
        "pubchem_molecular_weight": props.get("MolecularWeight"),
        "pubchem_xlogp": props.get("XLogP"),
        "pubchem_tpsa": props.get("TPSA"),
    }


def _pubchem_property_url(inchikey: str) -> str:
    properties = ",".join(
        [
            "Title",
            "IUPACName",
            "SMILES",
            "ConnectivitySMILES",
            "InChIKey",
            "MolecularFormula",
            "MolecularWeight",
            "XLogP",
            "TPSA",
        ]
    )
    return (
        f"{PUBCHEM_PUG_REST_BASE}/compound/inchikey/"
        f"{urllib.parse.quote(inchikey)}/property/{properties}/JSON"
    )


def _fetch_chembl_one(
    inchikey: str, *, url: str | None = None, timeout_seconds: float = 20
) -> dict[str, Any]:
    url = url or _chembl_molecule_url(inchikey)
    data = _get_json(url, timeout_seconds=timeout_seconds)
    molecules = data.get("molecules", [])
    chembl_ids = [
        molecule["molecule_chembl_id"]
        for molecule in molecules
        if molecule.get("molecule_chembl_id")
    ]
    first_id = chembl_ids[0] if chembl_ids else None
    return {
        "inchikey": inchikey,
        "chembl_status": "ok" if chembl_ids else "not_found",
        "chembl_ids": chembl_ids,
        "chembl_url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{first_id}/"
        if first_id
        else None,
    }


def _chembl_molecule_url(inchikey: str) -> str:
    query = urllib.parse.urlencode(
        {"molecule_structures__standard_inchi_key": inchikey, "limit": "5"}
    )
    return f"{CHEMBL_API_BASE}/molecule.json?{query}"


def _iter_enrichment_requests(service: str, requests: list[dict[str, Any]]):
    total = len(requests)
    with tqdm(
        requests,
        desc=f"{service} enrichment",
        total=total,
        unit="compound",
        dynamic_ncols=True,
    ) as progress:
        for request in progress:
            status = "cached" if request["cached"] else "download"
            url = request["url"] or "(cached)"
            progress.set_description(f"{service} {request['index']}/{total}")
            progress.set_postfix_str(f"status={status} url={url}", refresh=True)
            progress.write(
                f"{service} [{request['index']}/{total}] "
                f"status={status} url={url}"
            )
            yield request


def _rdkit_descriptors(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "canonical_smiles": None,
            "inchikey": None,
            "formula": None,
            "molecular_weight": None,
            "heavy_atom_count": None,
        }
    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "inchikey": Chem.MolToInchiKey(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
    }


def _get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_jsonl_cache(
    cache_path: str | Path | None, key_field: str
) -> dict[str, dict[str, Any]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get(key_field)
            if key:
                cache[key] = row
    return cache


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _loads_lmdb_record(value: bytes) -> dict[str, Any]:
    import pickle

    return pickle.loads(value)


def _require_polars():
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "polars is required to build the result DataFrame. Install it with "
            "`uv add polars --no-sync`, then run `uv sync --inexact` so the "
            "manual Uni-Core overlay is preserved."
        ) from exc
    return pl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DrugCLIP retrieval and build an enriched compounds table."
    )
    parser.add_argument("--drugclip-dir", default="external/DrugCLIP")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--mol-path", default=None)
    parser.add_argument("--pocket-path", default="data/2ie4.lmdb")
    parser.add_argument("--emb-dir", default=None)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--top-k", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-size-valid", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--enrich-pubchem", action="store_true")
    parser.add_argument("--pubchem-cache-path", default="data/pubchem_compounds.jsonl")
    parser.add_argument("--pubchem-limit", type=int, default=None)
    parser.add_argument("--enrich-chembl", action="store_true")
    parser.add_argument("--chembl-cache-path", default="data/chembl_compounds.jsonl")
    parser.add_argument("--chembl-limit", type=int, default=None)
    parser.add_argument(
        "--output-parquet",
        default="external/DrugCLIP/data/emb/ranked_compounds_enriched.parquet",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, cli_args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    drugclip_args = build_drugclip_args(
        drugclip_dir=cli_args.drugclip_dir,
        data_dir=cli_args.data_dir,
        checkpoint_path=cli_args.checkpoint_path,
        mol_path=cli_args.mol_path,
        pocket_path=cli_args.pocket_path,
        emb_dir=cli_args.emb_dir,
        results_path=cli_args.results_path,
        top_k=cli_args.top_k,
        batch_size=cli_args.batch_size,
        batch_size_valid=cli_args.batch_size_valid,
        num_workers=cli_args.num_workers,
        seed=cli_args.seed,
        fp16=not cli_args.no_fp16,
        cpu=cli_args.cpu,
    )

    smiles, scores = retrieve_mols_from_drugclip(drugclip_args)
    df = build_ranked_compounds_frame(
        smiles,
        scores,
        mol_lmdb_path=drugclip_args.mol_path,
        enrich_pubchem=cli_args.enrich_pubchem,
        pubchem_cache_path=cli_args.pubchem_cache_path,
        pubchem_limit=cli_args.pubchem_limit,
        enrich_chembl=cli_args.enrich_chembl,
        chembl_cache_path=cli_args.chembl_cache_path,
        chembl_limit=cli_args.chembl_limit,
    )

    output_parquet = Path(cli_args.output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_parquet)
    LOGGER.info("wrote %s", output_parquet)


if __name__ == "__main__":
    main()

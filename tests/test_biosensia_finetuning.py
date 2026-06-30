import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from biosensia_finetuning import (
    annotate_lmdb_records,
    choose_ligand_key,
    choose_pocket_key,
    target_fishing_rank_metrics,
)
from lmdb_helpers import read_lmdb_records, write_lmdb_records


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external/Uni-Core"))
sys.path.insert(0, str(REPO_ROOT / "external/DrugCLIP"))

from unimol.losses.cross_entropy import (  # noqa: E402
    _build_positive_mask,
    _multi_positive_direction_loss,
)
from unimol.tasks.drugclip import (  # noqa: E402
    LigandCenteredBatchDataset,
    _set_trainable_params,
)


def test_choose_identity_keys_prefers_metadata():
    record = {"smi": "CCO", "pocket": "1abc"}
    metadata = {
        "ligand_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "ligand_smiles": "CCO",
        "pocket": "uniprot:P12345",
        "pocket_geometry_hash": "pocketgeomsha1:abc",
    }

    assert (
        choose_ligand_key(record, metadata, policy="inchikey_or_smiles")
        == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    )
    assert (
        choose_pocket_key(record, metadata, policy="metadata_pocket")
        == "uniprot:P12345"
    )
    assert (
        choose_pocket_key(record, metadata, policy="geometry_hash")
        == "pocketgeomsha1:abc"
    )


def test_annotate_lmdb_records_adds_biosensia_keys(tmp_path):
    source = tmp_path / "train.lmdb"
    output = tmp_path / "annotated.lmdb"
    write_lmdb_records(
        [{"smi": "CCO", "pocket": "1abc", "atoms": ["C"], "coordinates": []}],
        source,
        overwrite=True,
        map_size=1 << 20,
    )

    summary = annotate_lmdb_records(
        source,
        output,
        split="train",
        pair_metadata={
            ("train", "0"): {
                "ligand_inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "ligand_smiles": "CCO",
                "pocket": "uniprot:P12345",
            }
        },
        map_size=1 << 20,
    )

    assert summary["records"] == 1
    assert summary["metadata_hits"] == 1
    record = read_lmdb_records(output)[0]
    assert record["ligand_key"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert record["pocket_key"] == "uniprot:P12345"
    assert record["smi"] == "CCO"


def test_target_fishing_rank_metrics_handles_multiple_positives():
    rankings = {
        "mol-a": ["p3", "p2", "p1"],
        "mol-b": ["p4", "p5"],
    }
    positives = {
        "mol-a": {"p1", "p2"},
        "mol-b": {"p5"},
    }

    metrics = target_fishing_rank_metrics(rankings, positives, top_k_values=(1, 2))

    assert metrics["queries"] == 2.0
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["top2_accuracy"] == 1.0
    assert metrics["recall_at_2"] == pytest.approx(0.75)
    assert metrics["mrr"] == pytest.approx(0.5)


def test_multi_positive_mask_and_loss_use_known_pairs():
    ligand_keys = ["lig-a", "lig-b", "lig-a"]
    pocket_keys = ["p1", "p2", "p3"]
    positives = {"lig-a": {"p1", "p3"}, "lig-b": {"p2"}}

    mask = _build_positive_mask(ligand_keys, pocket_keys, positives, torch.device("cpu"))

    assert mask.tolist() == [
        [True, False, True],
        [False, True, False],
        [True, False, True],
    ]

    logits = torch.tensor(
        [
            [4.0, 0.0, 3.0],
            [0.0, 5.0, 0.0],
            [3.0, 0.0, 4.0],
        ]
    )
    loss, count = _multi_positive_direction_loss(logits, mask)

    assert count.item() == 3
    assert loss.item() < 0.1


class _TinyDataset:
    can_reuse_epoch_itr_across_epochs = True

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_ligand_centered_batch_dataset_groups_repeated_ligands():
    dataset = LigandCenteredBatchDataset(
        _TinyDataset(),
        ["a", "a", "a", "b", "b", "c"],
        positives_per_ligand=2,
        seed=1,
    )
    dataset.set_epoch(1)

    batches = dataset.batch_by_size(np.arange(len(dataset)), batch_size=4)

    assert sorted(index for batch in batches for index in batch.tolist()) == list(range(6))
    assert any(
        sum(1 for index in batch.tolist() if index in {0, 1, 2}) >= 2
        for batch in batches
    )


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mol_model = torch.nn.Linear(2, 2)
        self.pocket_model = torch.nn.Linear(2, 2)
        self.mol_project = torch.nn.Linear(2, 2)
        self.pocket_project = torch.nn.Linear(2, 2)
        self.logit_scale = torch.nn.Parameter(torch.ones(1))


def test_projection_freeze_policy_only_trains_projection_heads():
    model = _DummyModel()

    _set_trainable_params(model, "projection")

    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {
        "mol_project.weight",
        "mol_project.bias",
        "pocket_project.weight",
        "pocket_project.bias",
    }

    _set_trainable_params(model, "projection-and-logit-scale")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert "logit_scale" in trainable

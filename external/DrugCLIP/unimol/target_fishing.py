#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""DrugCLIP target-fishing entry point.

The original ``retrieval.py`` implements virtual screening:

    query pocket(s) -> rank candidate molecules

Target fishing is the inverse workflow:

    query molecule(s) -> rank candidate pockets/targets

The model, checkpoint loading, Uni-Core argument handling, and scoring space are
the same. Only the task method changes from ``retrieve_mols`` to
``retrieve_pockets`` so the candidate side becomes pockets rather than
molecules.
"""

import logging
import os
import sys

import torch
from unicore import checkpoint_utils, distributed_utils, options, tasks


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.target_fishing")


def main(args):
    """Load DrugCLIP and rank candidate pockets for query molecules.

    ``args.mol_path`` is the target-fishing query LMDB. It should contain one or
    more molecule records with the same schema used by DrugCLIP retrieval:
    ``atoms``, ``coordinates``, and ``smi``.

    ``args.pocket_path`` is the candidate target/pocket LMDB. Its records should
    contain ``pocket``, ``pocket_atoms``, and ``pocket_coordinates``. Candidate
    pocket embeddings are saved under ``args.emb_dir`` by
    ``DrugCLIPTask.retrieve_pockets`` so future target-fishing runs against the
    same pocket library do not have to re-encode every pocket.
    """

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # Model setup is intentionally identical to ``retrieval.py``. This keeps
    # target-fishing scores comparable to virtual-screening scores because both
    # workflows use the same checkpoint and the same molecule/pocket projection
    # heads.
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    logger.info(args)
    model.eval()

    # ``retrieve_pockets`` returns pocket names and scores sorted from most to
    # least compatible with the query molecule set. If the query LMDB contains
    # multiple molecules, each pocket score is the maximum score over those
    # query molecules, mirroring the multi-pocket reduction in virtual
    # screening.
    names, scores = task.retrieve_pockets(
        model,
        args.mol_path,
        args.pocket_path,
        args.emb_dir,
        args.top_k,
    )

    os.makedirs(args.emb_dir, exist_ok=True)
    output_path = os.path.join(args.emb_dir, "ranked_pockets.txt")
    with open(output_path, "w") as f:
        for name, score in zip(names, scores):
            f.write(f"{name}\t{score}\n")
    logger.info("wrote ranked pockets to {}".format(output_path))


def cli_main():
    """Parse the same base arguments used by DrugCLIP retrieval."""

    parser = options.get_validation_parser()
    parser.add_argument(
        "--mol-path",
        type=str,
        default="",
        help="path for query molecule data",
    )
    parser.add_argument(
        "--pocket-path",
        type=str,
        default="",
        help="path for candidate pocket data",
    )
    parser.add_argument(
        "--emb-dir",
        type=str,
        default="",
        help="path for saved candidate-pocket embedding data",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help="number of top-ranked pockets to write",
    )
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()

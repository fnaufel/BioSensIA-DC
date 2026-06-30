#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility entry point for BioSensIA-DC target fishing.

The target-fishing implementation lives in the repository-root module
``biosensia_target_fishing``. This file keeps a DrugCLIP-repository-style
``python ./unimol/target_fishing.py ...`` command working for scripts that run
from ``external/DrugCLIP``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

from unicore import distributed_utils, options


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biosensia_target_fishing import target_fishing_main  # noqa: E402


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

    distributed_utils.call_main(args, target_fishing_main)


if __name__ == "__main__":
    cli_main()

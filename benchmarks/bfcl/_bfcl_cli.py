"""Thin wrapper around BFCL's CLI that registers google/gemma-4-12B-it.

Gemma-4 isn't an officially supported BFCL model, and the only gate that matters
is `MODEL_CONFIG_MAPPING` (the `SUPPORTED_MODELS` list is a convenience index).
So instead of editing the installed `bfcl_eval` package we inject the config here
-- mutating the mapping in place, which every module that did
`from ...model_config import MODEL_CONFIG_MAPPING` sees, since it's the same dict
object -- then delegate to BFCL's normal Typer entry point.

Run exactly like `bfcl`, e.g.:
    python benchmarks/bfcl/_bfcl_cli.py generate --model google/gemma-4-12B-it ...
    python benchmarks/bfcl/_bfcl_cli.py evaluate --model google/gemma-4-12B-it ...
"""
import os
import sys

# Import the sibling handler regardless of how this file is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bfcl_eval.constants import model_config as _mc
from gemma4_handler import Gemma4Handler

_MODEL_ID = "google/gemma-4-12B-it"

if _MODEL_ID not in _mc.MODEL_CONFIG_MAPPING:
    _mc.MODEL_CONFIG_MAPPING[_MODEL_ID] = _mc.ModelConfig(
        model_name=_MODEL_ID,
        display_name="Gemma-4-12b-it (Prompt)",
        url="https://ai.google.dev/gemma/docs/core",
        org="Google",
        license="gemma-terms-of-use",
        model_handler=Gemma4Handler,
        input_price=None,
        output_price=None,
        is_fc_model=False,      # prompt mode -- functions rendered into the prompt
        underscore_to_dot=False,
    )

from bfcl_eval.__main__ import cli

if __name__ == "__main__":
    cli()

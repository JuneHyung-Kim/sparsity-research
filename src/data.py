"""WikiText-2 evaluation stream (SparseGPT/Wanda protocol for comparable PPL)."""
from datasets import load_dataset

_WIKITEXT = "Salesforce/wikitext"  # bare "wikitext" id is gone in datasets>=3


def get_wikitext2_testenc(tokenizer):
    test = load_dataset(_WIKITEXT, "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    return enc.input_ids

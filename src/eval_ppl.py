"""WikiText-2 perplexity (SparseGPT/Wanda averaging convention)."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_ppl(model, testenc, seqlen, device, chunk=256):
    """Mean CE per window is scaled by seqlen (SparseGPT convention), so numbers
    stay comparable with the earlier LLaMA-2 sweeps. CE is computed in `chunk`-
    token slices: upcasting a full [seqlen, vocab] logits row to fp32 costs ~2GB
    on a 262k-vocab model (Gemma), which OOMs a 24GB card that is already full
    of offloaded weights."""
    model.eval()
    nsamples = testenc.shape[1] // seqlen
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        ntok = shift_labels.numel()
        ce_sum = 0.0
        for j in range(0, shift_labels.shape[1], chunk):
            sl = shift_logits[:, j:j + chunk, :].float()
            lb = shift_labels[:, j:j + chunk]
            ce_sum += F.cross_entropy(sl.reshape(-1, sl.shape[-1]),
                                      lb.reshape(-1), reduction="sum").item()
        nlls.append(ce_sum / ntok * seqlen)
    return math.exp(sum(nlls) / (nsamples * seqlen))

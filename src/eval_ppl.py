"""WikiText-2 perplexity (SparseGPT/Wanda averaging convention)."""
import math

import torch
import torch.nn as nn


@torch.no_grad()
def eval_ppl(model, testenc, seqlen, device):
    model.eval()
    nsamples = testenc.shape[1] // seqlen
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1))
        nlls.append(loss.item() * seqlen)
    return math.exp(sum(nlls) / (nsamples * seqlen))

import torch 
from util import kl_anneal_function

def loss_fn(NLL,logp, target, length, mean, logv, anneal_function, step, k0, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k0, x0)

        return NLL_loss, KL_loss, KL_weight

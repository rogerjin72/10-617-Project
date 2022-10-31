import torch

epsilon = 1e-8


def NT_xent_loss(outputs, temperature=1.0):
    """
    Compute NT xent loss for output representations

    Parameters
    ----------
    outputs : torch.Tensor
        model representations
    temperature : float, optional
        temperature parameter for loss, default 1.0
    
    Returns
    -------
    torch.Tensor
        NT xent loss
    """
    # compute similarity
    batch_size = outputs.shape[0]
    normed = outputs / (outputs.norm(dim=1).view(batch_size,1) + epsilon)
    sim = normed @ normed.T / temperature

    # get logits
    sim_exp = torch.exp(sim)
    sim_exp = sim * (1 - torch.eye(batch_size, batch_size))
    logits = sim_exp / (torch.sum(sim_exp, dim=1).view(batch_size, 1) + epsilon)
    
    n = batch_size // 2

    # get loss of i-th item with i-th target and vice versa
    NT_xent_loss = - torch.log(logits + epsilon)
    loss = NT_xent_loss[n:, :n].diag().sum() + NT_xent_loss[:n, n:].diag().sum()
    loss  = loss / batch_size

    return loss

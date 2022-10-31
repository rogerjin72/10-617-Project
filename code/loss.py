import torch

epsilon = 1e-8


def NT_xent_loss(outputs, temperature=1.0, slices=2):
    """
    Compute NT xent loss for output representations

    Parameters
    ----------
    outputs : torch.Tensor
        model representations
    temperature : float, optional
        temperature parameter for loss, default 1.0
    slices : int, optional
        number of slices for the dataset

    Returns
    -------
    torch.Tensor
        NT xent loss
    """
    batch_size = outputs.shape[0]

    if batch_size % slices:
        raise AssertionError('number of slices should divide number of inputs')

    # compute similarity
    normed = outputs / (outputs.norm(dim=1).view(batch_size,1) + epsilon)
    sim = normed @ normed.T / temperature

    # get logits
    sim_exp = torch.exp(sim)
    sim_exp = sim * (1 - torch.eye(batch_size, batch_size))
    logits = sim_exp / (torch.sum(sim_exp, dim=1).view(batch_size, 1) + epsilon)
    
    n = batch_size // slices

    # get loss of i-th item with i-th target and vice versa
    NT_xent_loss = - torch.log(logits + epsilon)
    losses = 0

    for i in range(slices):
        for j in range(slices):
            if i != j:
                loss = NT_xent_loss[i * n: (i + 1) * n, j * n: (j + 1) * n].diag()
                losses += loss.sum()
    
    return losses / batch_size

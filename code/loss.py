import torch

epsilon = 1e-8


def NT_xent_loss(outputs, temperature=1.0, slices=2, rince=False):
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
    sim_exp = sim_exp * (1 - torch.eye(batch_size, batch_size)).cuda()
    logits = sim_exp / (torch.sum(sim_exp, dim=1).view(batch_size, 1) + epsilon)
    
    n = batch_size // slices

    # get loss of i-th item with i-th target and vice versa
    if rince == False:
        NT_xent_loss = - torch.log(logits + epsilon)
        losses = 0

        for i in range(slices):
            for j in range(slices):
                if i != j:
                    loss = NT_xent_loss[i * n: (i + 1) * n, j * n: (j + 1) * n].diag()
                    losses += loss.sum()
        losses = losses / batch_size
    else:
        q = 0.1
        lam = 0.01
        pos = None
        neg = None
        # NT_xent_loss = - (logits) / q
        losses = 0
        
        for i in range(slices):
            row_pos = None
            row_neg = None
            for j in range(slices):
                if i != j:
                    sim_mat = logits[i * n: (i + 1) * n, j * n: (j + 1) * n]
                    pos_mat = sim_mat.diag()
                    neg_mat = torch.sum(sim_mat * (1 - torch.eye(sim_mat.shape[0], sim_mat.shape[1])).cuda(), 1)
                    if row_pos is None:
                        row_pos = pos_mat
                    else:
                        row_pos += pos_mat
                    if row_neg is None:
                        row_neg = neg_mat
                    else:
                        row_neg += neg_mat
                    # pos += sim_mat.diag()
                    # neg += torch.sum()
                    # loss = NT_xent_loss[i * n: (i + 1) * n, j * n: (j + 1) * n].diag()
                    # losses += loss.sum()
            if pos is None:
                pos = row_pos
            else:
                pos = torch.cat((pos, row_pos))
            if neg is None:
                neg = row_neg
            else:
                neg = torch.cat((neg, row_neg))
        neg = ((lam*(pos + neg))**q) / q
        pos = -(pos**q) / q
        losses = pos.mean() + neg.mean()
    return losses

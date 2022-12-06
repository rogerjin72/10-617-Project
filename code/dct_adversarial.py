import torch
import numpy as np
import torch.nn.functional as F 

from loss import NT_xent_loss
from adversarial import project, perturb_img

def dct(im: torch.Tensor):
    """
    Type II discrete cosine transform

    Parameters
    ----------
    im : 
        input image, shape B, C, H, W
    
    Returns
    -------
    torch.Tensor:
        transformed image using DCT
    """
    b, c, h, w = im.shape

    # get indices
    cols = torch.arange(h).repeat(h).reshape(h, h)
    rows = cols.T
    index = rows * (2 * cols + 1) * np.pi / 2 / h
    
    # create dct matrix
    dct_mat = np.sqrt(2 / h) * torch.cos(index)
    dct_mat[0, :] = dct_mat[0, : ] / np.sqrt(2)
    dct_mat = dct_mat.cuda()
    
    # apply transformation
    transformed = torch.matmul(im, dct_mat.T)
    transformed = torch.matmul(dct_mat, transformed)
    
    return transformed


def inv_dct(im):
    """
    Type III discrete cosine transform

    Parameters
    ----------
    im : 
        input image, shape B, C, H, W
    
    Returns
    -------
    torch.Tensor:
        transformed image using inverse DCT
    """
    b, c, h, w = im.shape

    # get indices
    cols = torch.arange(h).repeat(h).reshape(h, h)
    rows = cols.T
    index = cols * (2 * rows + 1) * np.pi / 2 / h

    # create inverse dct matrix
    inv_dct_mat = np.sqrt(2 / h) * torch.cos(index)
    inv_dct_mat[:, 0] = 1.0 / np.sqrt(h)
    inv_dct_mat.requires_grad = True
    inv_dct_mat = inv_dct_mat.cuda()

    # apply transformation
    transformed = torch.matmul(im, inv_dct_mat.T)
    transformed = torch.matmul(inv_dct_mat, transformed)
    return transformed


class DCTFGSM(object):
    """
    class to handle fast gradient sign adversarial attacks in dct space
    """
    def __init__(self, model: torch.nn.Module, linear: torch.nn.Module, epsilon: float, alpha: float, n: int):
        """
        Parameters
        ----------
        model :
            pretrained model 
        linear : 
            linear classifier
        epsilon :
            maximum magnitude of perturbation
        alpha: 
            learning rate
        n :
            number of iterations

        Returns
        -------
        None
        """
        self.model = model
        self.linear = linear
        self.epsilon = epsilon
        self.alpha = alpha
        self.n = n


    def get_adversarial_example(self, imgs: torch.Tensor, labels: torch.Tensor=None, perturb: bool=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        labels : optional
            labels for images, default None
        perturb : optional
            adds random noise to images, default True

        Returns
        -------
        torch.Tensor
            original image perturbed by the fast gradient sign method
        """
        adv = imgs.clone()
        
        if labels is None:
            labels = self.model(imgs).argmax(axis=1)

        if perturb:
            adv = perturb_img(adv, self.epsilon, 0, 256)
        
        adv = dct(adv)
        orig = adv.clone()

        # apply gradients
        adv.requires_grad = True

        self.model.eval()
        with torch.enable_grad():
            for _ in range(self.n):
                self.model.zero_grad()
                self.linear.zero_grad()
                
                inp = inv_dct(adv)

                # compute loss
                logits = self.model(inp)
                logits = self.linear(logits)
                loss = F.cross_entropy(logits, labels)

                # get gradient of loss for image
                grad = torch.autograd.grad(loss, adv, only_inputs=True, retain_graph=False)
                grad = torch.sign(grad[0])

                # step
                adv = adv + self.alpha * grad
                adv = project(adv, orig, self.epsilon)
    
        adv_img = inv_dct(adv)
        return adv_img.detach()

class DCTAdversary(object):
    def __init__(self, model: torch.nn.Module, epsilon: float, alpha: float, n: int, temperature=1.0):
        """
        Parameters
        ----------
        model :
            model
        linear : 
            linear classifier for model
        epsilon :
            maximum magnitude of perturbation
        alpha: 
            learning rate
        n :
            number of iterations
        temperature : float, optional
            temperature parameters for logits, default 1.0

        Returns
        -------
        None
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.n = n
        self.temperature = temperature

    def get_adversarial_example(self, imgs: torch.Tensor, target: torch.Tensor, perturb=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        target : 
            target images
        perturb : bool, optional
            adds random noise to images, default True

        Returns
        -------
        torch.Tensor
            original image perturbed in the dct space by gradient of loss wrt target samples
        """
        # copy inputs and apply perturbation
        adv = imgs.clone()
        if perturb:
            adv = perturb_img(adv, self.epsilon, 0, 256)

        adv = dct(adv)
        orig = adv.clone()

        # apply gradients
        adv.requires_grad = True

        self.model.eval()
        with torch.enable_grad():
            for _ in range(self.n):
                self.model.zero_grad()
                # get loss
                inp = inv_dct(adv)
                inp.retain_grad()

                logits = self.model(torch.cat([inp, target]))
                loss = NT_xent_loss(logits, self.temperature)

                # get gradient of loss for image in dct space
                grad = torch.autograd.grad(loss, adv, only_inputs=True, retain_graph=False)
                grad = torch.sign(grad[0])

                # step
                adv = adv + self.alpha * grad
                adv = project(adv, orig, self.epsilon)

        adv_img = inv_dct(adv)
        return adv_img.detach()

    def get_adversarial_loss(self, imgs: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, perturb=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        target : 
            target images
        optimizer : 
            optimizer for training
        perturb : bool, optional
            adds random noise to images, default True

        Returns
        -------
        torch.Tensor
            loss of the adversarial input wrt target samples
        """
        # get adversarial examples
        adv = self.get_adversarial_example(imgs, target, perturb)
        
        self.model.train()
        optimizer.zero_grad()
        
        # get loss
        logits = self.model(torch.cat([adv, target]))
        loss = NT_xent_loss(logits, self.temperature)
        return loss
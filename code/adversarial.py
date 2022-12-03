import torch
import torch.nn.functional as F 

from loss import NT_xent_loss


def project(adv: torch.Tensor, orig: torch.Tensor, epsilon: float):
    """
    constrains perturbation to within epsilon ball of the original input

    Parameters
    ----------
    adv : 
        adversarially perturbed images
    orig : 
        original images
    epsilon :
        radius of ball

    Return
    ------
    torch.Tensor : 
        bounded adversarial inputs
    """
    # get epsilon bounds for original image
    max_x = orig + epsilon
    min_x = orig - epsilon

    prev = adv.clone()

    # bound adversarial input
    adv = torch.max(torch.min(adv, max_x), min_x)
    return adv


def perturb_img(imgs: torch.Tensor, epsilon: float, min_val: float, max_val: float):
    """
    add noise to image

    Parameters
    ----------
    imgs :
        input images
    epsilon :
        max absolute value of noise
    min_val : 
        minimum pixel value
    max_val :
        maximum pixel value
    
    Returns
    -------
    torch.Tensor
    """
    noise = torch.zeros(imgs.shape).uniform_(-epsilon, epsilon).cuda()
    imgs += noise
    return imgs.clamp(min_val, max_val)


class FGSM(object):
    """
    class to handle fast gradient sign adversarial attacks
    """
    def __init__(self, model: torch.nn.Module, epsilon: float, min_val: float, max_val: float, alpha: float, n: int):
        """
        Parameters
        ----------
        model :
            model with linear classifier
        epsilon :
            maximum magnitude of perturbation
        alpha: 
            learning rate
        min_val : 
            minimum pixel value
        max_val :
            maximum pixel value
        n :
            number of iterations

        Returns
        -------
        None
        """
        self.model = model
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.n = n


    def get_adversarial_example(self, imgs: torch.Tensor, labels: torch.Tensor=None, perturb=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        labels : optional
            labels for images, default None
        perturb : bool, optional
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
            adv = perturb_img(adv, self.epsilon, self.min_val, self.max_val)
        
        # apply gradients
        adv.requires_grad = True

        self.model.eval()
        with torch.enable_grad():
            for _ in range(self.n):
                self.model.zero_grad()
                # compute loss
                logits = self.model(adv)
                loss = F.cross_entropy(logits, labels)

                # get gradient of loss for image
                grad = torch.autograd.grad(loss, adv, only_inputs=True, retain_graph=False)
                grad = torch.sign(grad[0])

                # step
                adv = adv + self.alpha * grad
                adv = adv.clamp(self.min_val, self.max_val)
                adv = project(adv, imgs, self.epsilon)
    
        return adv.detach()


class InstanceAdversary(object):
    """
    class to handle instance-wise adversarial attacks
    """
    def __init__(self, model: torch.nn.Module, epsilon: float, alpha: float, min_val: float, max_val: float, n: int, temperature=1.0):
        """
        Parameters
        ----------
        model :
            model with linear classifier
        epsilon :
            maximum magnitude of perturbation
        alpha: 
            learning rate
        min_val : 
            minimum pixel value
        max_val :
            maximum pixel value
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
        self.min_val = min_val
        self.max_val = max_val
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
            original image perturbed by gradient of loss wrt target samples
        """
        # copy inputs and appy perturbation
        adv = imgs.clone()
        if perturb:
            adv = perturb_img(adv, self.epsilon, self.min_val, self.max_val)

        # apply gradients
        adv.requires_grad = True

        self.model.eval()
        with torch.enable_grad():
            for _ in range(self.n):
                self.model.zero_grad()
                # get loss
                logits = self.model(torch.cat([adv, target]))
                loss = NT_xent_loss(logits, self.temperature)

                # get gradient of loss for image
                grad = torch.autograd.grad(loss, adv, only_inputs=True, retain_graph=False)
                grad = torch.sign(grad[0])

                # step
                adv = adv + self.alpha * grad
                adv = adv.clamp(self.min_val, self.max_val)
                adv = project(adv, imgs, self.epsilon)
    
        return adv.detach()

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

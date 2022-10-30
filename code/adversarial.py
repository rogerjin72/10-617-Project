import torch
import torch.nn.functional as F 


def project(adv: torch.Tensor, orig: torch.Tensor, epsilon: float):
    max_x = orig + epsilon
    min_x = orig - epsilon

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
    noise = torch.Tensor(imgs.shape).uniform(-epsilon, epsilon)
    imgs += noise
    return imgs.clamp(min_val, max_val)


class FGSM(object):
    """
    class to handle fast gradient sign adversarial attacks
    """
    def __init__(model: torch.nn.Module, epsilon: float, min_val: float, max_val: float, alpha: float, n: int):
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


    def get_adversarial_example(self, imgs: torch.Tensor, labels: torch.Tensor, perturb=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        labels : 
            labels for images
        perturb : bool, optional
            adds random noise to images, default True

        Returns
        -------
        torch.Tensor
            original image perturbed by the fast gradient sign method
        """
        adv = imgs.clone()

        if perturb:
            adv = perturb_img(adv, epsilon, min_val, max_val)
        
        # apply gradients
        adv.requires_grad = True

        self.model.eval()
        with torch.enable_grad():
            for _ in range(n):
                self.model.zero_grad()

                # compute loss
                logits = self.model(x)
                loss = F.cross_entropy(logits, labels)

                # get gradient of loss for image
                grad = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)
                grad = torch.sign(grad[0])

                # step
                adv += self.alpha * grad
                adv = adv.clamp(min_val, max_val)
                adv = project(adv, imgs, self.epsilon)
    
    return adv.detach()


class InstanceAdversary(object):
    """
    class to handle instance-wise adversarial attacks
    """
    def __init__(model: torch.Tensor, epsilon: float, alpha: float, min_val: float, max_val: float, n: int):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.n = n
    
    def get_adversarial_example(self, imgs: torch.Tensor, target: torch.Tensor, perturb=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        labels : 
            labels for images
        perturb : bool, optional
            adds random noise to images, default True

        Returns
        -------
        torch.Tensor
            original image perturbed by gradient of loss wrt target samples
        """
        adv = imgs.clone()

        if perturb:
            adv = perturb_img(adv, epsilon, min_val, max_val)

        self.model.eval()
        with torch.enable_grad():
            for _ in range(n):
                self.model.zero_grad()
                # get loss
                loss = 1-F.cosine_similarity(self.model(adv), self.model(target)).mean()
                
                # get gradient of loss for image
                grad = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)
                grad = torch.sign(grad[0])

                # step
                adv += self.alpha * grad
                adv = adv.clamp(min_val, max_val)
                adv = project(adv, imgs, self.epsilon)
    
    return adv.detach()

    def get_adversarial_loss(self, imgs: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, perturb=True):
        """
        Parmeters
        ---------
        imgs :
            input images
        labels : 
            labels for images
        optimizer : 
            optimizer for training
        perturb : bool, optional
            adds random noise to images, default True

        Returns
        -------
        torch.Tensor
            loss of the adversarial input wrt target samples
        """
        adv = self.get_adversarial_example(imgs, target, perturb)
        
        self.model.train()
        optimizer.zero_grad()
        
        batch_size = imgs.shape[0]
        sim = F.cosine_similarity(model(adv), model(target)).sum() / batch_size

        return 1 - sim

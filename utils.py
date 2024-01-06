import math
import torch
import os

def makedir(dir):
    """
    Creates a directory if it does not exist.

    Args:
        dir (str): The directory to be created.
    
    Returns:
        None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def cosine_alpha_bar(t, s=0.008):
        return math.cos((t+s)/(1+s)*math.pi/2)**2
    
def cosine_beta_schedule(num_timesteps, beta_min, beta_max, s=0.008):
    beta = []
    for t in range(num_timesteps):
        t1 = t/num_timesteps
        t2 = (t+1)/num_timesteps
        alpha_bar_1 = cosine_alpha_bar(t1,s=0.008)
        alpha_bar_2 = cosine_alpha_bar(t2,s=0.008)
        beta.append(beta_min + min(beta_max-beta_min, 1-alpha_bar_2/alpha_bar_1))
    return torch.Tensor(beta)
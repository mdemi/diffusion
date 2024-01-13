# Diffusion
A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPMs) with a U-Net architecture.

![galaxies](https://github.com/mdemi/diffusion/assets/34105945/c20726c3-c279-4574-9269-49c78180d302)
![particles](https://github.com/mdemi/diffusion/assets/34105945/5e123100-14ec-45a0-b322-859c037621f2)
![triangulations](https://github.com/mdemi/diffusion/assets/34105945/b1de6f7d-2b00-423d-ada5-0fcb2e0bc13a)

We train diffusion models on three datasets:

1- Galaxies: Images of galaxies obtained from the Sloan Digital Sky Survey,

2- Particles: 2D projections of simulated energy deposition patterns of particles in liquid argon,

3- Triangulations: Fine Regular triangulations of an 8x8 grid,

and show that the same architecture performs well on these three very different datasets with minimal hyperparameter tuning.

## References

### DDPMs:

https://arxiv.org/abs/2006.11239

### U-Net:

https://arxiv.org/abs/1505.04597

### Galaxies:

https://zenodo.org/records/3565489#.Y3vFKS-l0eY

https://academic.oup.com/mnras/article/435/4/2835/1022913

### Particles:

https://indico.slac.stanford.edu/event/7540/

https://github.com/makagan/SSI_Projects/tree/main

### Triangulations:

https://cy.tools/


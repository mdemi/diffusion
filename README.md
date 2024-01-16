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

## Datasets
### Galaxies

243,434 images of galaxies from the Galaxy Zoo 2 dataset, taken by the Sloan Digital Sky Survey.

### Particles

2D projections of the energy deposition patterns of four types of particles (electrons, photons, muons and protons) simulated in liquid argon medium.

The dataset is taken from the 51st SLAC Summer Institute (SSI 2023).

### Triangulations

Fine regular triangulations of an 8x8 grid.

Regular triangulations in higher dimensions play a crucial role in constructing Calabi-Yau manifolds and the associated solutions of string theory. This toy example is a first step towards generating Calabi-Yau manifolds with diffusion models.

A triangulation of a set of points is called regular if it can be generated as follows. Lift every point into one higher dimension, where a height is assigned to each point. Take the convex hull of the resulting set of points and project down the downward-facing faces of this polyhedron to obtain a triangulation. Not all triangulations can be obtained this way, i.e., not every triangulation is regular.

A triangulation of a set of points is called fine if every point is included in the triangulation.

In this dataset, for every regular triangulation we generate a 2D image where the values of the pixels are the heights of the points. Then, we train the diffusion model on these images. For inference, we use the diffusion model to generate a 2D image of heights and use CYTools (https://cy.tools/) to convert them into triangulations.

For more information about fine regular triangulations and their importance in string theory, see e.g. https://arxiv.org/abs/2211.03823 or https://arxiv.org/abs/2008.01730.

## References

### DDPMs:

https://arxiv.org/abs/2006.11239

### U-Net:

https://arxiv.org/abs/1505.04597

### Galaxies:

https://zenodo.org/records/3565489#.Y3vFKS-l0eY

https://academic.oup.com/mnras/article/435/4/2835/1022913

https://academic.oup.com/mnras/article/461/4/3663/2608720?login=false

https://www.zooniverse.org/projects/zookeeper/galaxy-zoo

### Particles:

https://indico.slac.stanford.edu/event/7540/

https://github.com/makagan/SSI_Projects/tree/main

### Triangulations:

https://cy.tools/

https://arxiv.org/abs/2211.03823

https://arxiv.org/abs/2008.01730




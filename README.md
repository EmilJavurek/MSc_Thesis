# Going with the Flow: Variational Bayesian Inference using Flow Matching

Code repository for my Master's dissertation project.

## Abstract

In this dissertation, we investigate Flow Matching for Posterior Estimation (FMPE) as the latest method for variational inference. FMPE learns a Continuous Normalizing Flow (CNF) from a simple and easy-to-sample source distribution to an approximation of the posterior. The method has been shown to achieve state-of-the-art performance in several generative as well as inference tasks. We focused on the ability of FMPE to learn a multimodal posterior, showing good performance in a toy example of a mixture of Gaussians and sub-optimal performance on a real radiocarbon dating example. Further work should assess a wider variety of path designs at higher computational budgets to better explore the limitations of FMPE.

## Installation Guide

The project is run on Python version **3.9.13**.
To set up the Python environment for this project, follow the steps below:

 - 1. Create a new Conda environment named `fmpe_env`
 - 2. Update it with the required dependencies using the `environment.yml` file
 - 3. Install the Jupyter kernel for the newly created environment

```bash
conda create --name fmpe_env
conda env update --name fmpe_env -f environment.yml
python -m ipykernel install --user --name fmpe_env --display-name "(fmpe_env)"
```

### License

The repository is published under the MIT License.
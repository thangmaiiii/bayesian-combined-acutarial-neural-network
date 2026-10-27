This folder contains codes that I've used for my thesis. Mainly, the files are divided into 3 broad groups: Python files, Python notebooks, and folders containing support files.

Notebook files are used for training, testing, and running codes. I prefer running code in notebooks as I can run cell by cell and see the output, making it easier to debug. The details are as follows:

- coefficients_interpret.ipynb: contains the extraction of gamma's alpha parameters under GLM, VI, HMC-NUTS, NeuTra.

- data_clean.ipynb: cleaning data for model uses.

- gamma_bcann_hmc_nuts.ipynb: Run HMC-NUTS for Gamma BCANN with Gaussian prior.

- gamma_bcann_horseshoe_neutra.ipynb: Run NeuTra for Gamma BCANN with regularised horseshoe prior.

- gamma_bcann_horseshoe.ipynb: Perform variational inference on Gamma BCANN with regularised horseshoe prior.

- gamma_bcann_neutra.ipynb: Run Neutra for GammaBCANN with Gaussian prior.

- gamma_bcann_vi.ipynb: Perform variational inference on Gamma BCANN with Gaussian prior + Gamma CANN training and GLM fitting.

- gamma_hmc_neutra_summary.ipynb: Aggregtates Gamma HMC NUTS annd Neutra samples, and perform diagnostics on them, and some predictions too.

- gamma_horseshoe_hmc.ipynb:  Run HMC-NUTS for Gamma BCANN with regularised horseshoe prior.

- poisson_bcann_hmc_nuts.ipynb: Run HMC-NUTS for Poisson BCANN with Gaussian prior.

- poisson_bcann_horseshoe_hmc.ipynb: Run HMC-NUTS for Poisson BCANN with regularised horseshoe prior.

- poisson_bcann_horseshoe.ipynb: Perform variational inference on Poisson BCANN with regularised horseshoe prior.

- poisson_bcann_vi.ipynb: Perform variational inference on Poisson BCANN with Gaussian prior + Poisson CANN training and GLM fitting.

- poisson_hmc_neutra_summary.ipynb: Aggregtates Poisson HMC NUTS annd Neutra samples, and perform diagnostics on them, and some predictions too.

- poisson_horseshoe_neutra.ipynb: Run NeuTra for Poisson BCANN with regularised horseshoe prior.

- poisson_neutra.ipynb: Run NeuTra for Poisson BCANN with Gaussian prior.

Note that in these notebooks, some have their device set to "cuda", which is NVIDIA GPU's, others have their device set to "cpu", which are ran locally. The device setting is torch.set_default_device("device_name").

Python files contain model classes and functions that are backbone to the implement of BCANN in this thesis:

- bcann_neutra.py: Contain functions to run NeuTra.

- codebase_0.py: Contain Model classes (MLP, CANN, BayesianMLP, BayesianCANN).

- gamma_bcann_hmc_nuts.py: Contain functions to run NUTS for GammaBCANN.

- gamma_bcann_vi.py: Contain GammaBCANN and GammaCANN model classes and functions to run variational inference on these models.

- gamma_horseshoe.py: Contain HorseshoeGammaBCANN and functions to run VI on HorseshoeGammaBCANN.

- gamma_nuts_parallel.py: Contain functions to run NUTS parallel on GPU for Gamma classes to save time and utilise computation resources.

- glm_interpretability.py: Contain some coefficents diagnostics functions for GLM, CANN, BCANN (mostly unused).

- poisson_bcann_hmc_nuts.py: Contain functions to run NUTS for PoissonBCANN.

- poisson_bcann_vi.py: Contain PoissonBCANN and PoissonBCANN model classes and functions to run variational inference on these models.

- poisson_horseshoe.py: Contain HorseshoePoissonBCANN and functions to run VI on HorseshoePoissonBCANN.

- poisson_nuts_parallel.py: Contain functions to run NUTS parallel on GPU for Poisson classes to save time and utilise computation resources.

In these files, each function would have a near identical counterpart, one with "gamma_" prefix and one without (i.e build_bcann_guide and gamma_build_bcann_guide). The one without is the function for Poisson and the one with "gamma_" prefix is used for Gamma. Most functions and classes have docstrings detailing its uses and requirements.

"requirements.txt" contains the version of required packages to run this codebase, and "requirements-mac.txt" contains the version of required packages to run this codebase on Mac (without CUDA).

The folders , "gamma_neutra_chains", "gamma_nuts_chains", "gamma_nuts_chains_horseshoe", "neutra_chains_poisson_horseshoe", "neutra_chains_gamma_horseshoe", "poisson_neutra_chains", "poisson_nuts_chains", "poisson_nuts_chains_horseshoe" contain separate chains from NUTS and Neutra implementaions, saved here for safekeeping and later uses.

The folders "runs" contains some VI runs, "hmc_results" contains aggregated samples from NUTS and NeuTra implementaions.

AI disclaimer: This codebase is built with the assistance of Claude Code.
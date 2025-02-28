# Global Framework for Simultaneous Emulation Across the Nuclear Landscape
# BANANNE

Despite recent advancements in many-body methods and nucleon-nucleon interactions derived from chiral effective field theory, performing accurate nuclear many-body calculations with quantifiable uncertainties remains a significant challenge for medium and heavy nuclei. To address this, we introduce a hierarchical framework that combines ab initio many-body calculations with a Bayesian neural network, developing emulators capable of accurately predicting nuclear properties across the nuclear chart. This approach enables the rapid evaluation of nuclear properties across multiple isotopes simultaneously. We benchmark our developments using the oxygen isotopic chain, achieving precise results for ground-state energies and nuclear charge radii, while providing robust uncertainty quantification. Our method demonstrates robust extrapolation and interpolation capabilities to nuclei outside of the training data, making it highly effective for transfer learning and enabling predictions in extreme regions of the nuclear chart. The emulator's flexible architecture lays the foundation for enabling global uncertainty quantification across the nuclear landscape, as well as guiding searches at rare isotope beam facilities towards isotopes that will reveal the most about the underlying nuclear forces.

See the full paper [here](https://arxiv.org/abs/2502.20363).


This repository contains the code and data for reproducing the results in the paper 'Global Framework for Simultaneous Emulation Across the Nuclear Landscape'.

### Features

- *Hierarchical Bayesian Framework*: Incorporates multi-fidelity data and enables uncertainty quantification.

- *Efficient Emulation*: Achieves accurate predictions at a fraction of the computational cost of full ab initio methods.

- *Simultaneous Predictions*: Capable of predicting nuclear properties across multiple isotopes, capturing correlations within isotopic chains.

- *Extrapolation Capabilities*: Performs zero-shot predictions beyond training data, making it ideal for rare isotope searches.
Transfer Learning: The emulator’s flexible architecture allows for integration into other nuclear physics models.

## Installation

To install the code, clone the repository and install the required dependencies using pip:

```
git clone
cd bannane
pip install -r requirements.txt
```
Ensure that you have Python 3.8+ installed along with PyTorch and Pyro for Bayesian inference.

## Structure

The repository is structured as follows:
```
bannane/
│
├── data/                   # Dataset files
│   ├── all_o/              # Oxygen isotopic chain data used for training, evaluation, and testing
│
├── notebooks/              # Jupyter notebooks for exploration and benchmarking
│   ├── simple/             # Example training and benchmarking notebook
│   ├── studies/            # In-depth analysis of model performance
│
├── results/                # Model outputs for reproducibility
│   ├── predictions/        # Predicted nuclear properties and corresponding uncertainty quantification
│   ├── sensitivity/        # Sensitivity analysis results
│
├── scripts/                # Scripts for training, evaluation, and inference
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Performance evaluation and benchmarking
│   ├── slurm/              # SLURM job scripts for cluster-based training
│
├── models/                 # Pre-trained models and checkpoint files
│
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── LICENSE                 # MIT License

```


## License

This code is released under the MIT License.

## Authors

Please cite:
```bibtex
@misc{belley2025globalframeworksimultaneousemulation,
      title={Global Framework for Simultaneous Emulation Across the Nuclear Landscape},
      author={Antoine Belley and Jose M. Munoz and Ronald F. Garcia Ruiz},
      year={2025},
      eprint={2502.20363},
      archivePrefix={arXiv},
      primaryClass={nucl-th},
      url={https://arxiv.org/abs/2502.20363},
}

```

EMA-LAB @ MIT
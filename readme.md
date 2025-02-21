# Global Framework for Simultaneous Emulation Across the Nuclear Landscape
# BANANNE

### Abstract

Despite recent advancements in many-body methods and nucleon-nucleon interactions derived from chiral effective field theory, performing accurate nuclear many-body calculations with quantifiable uncertainties remains a significant challenge for medium and heavy nuclei. To address this, we introduce a hierarchical framework that combines ab initio many-body calculations with a Bayesian neural network, developing emulators capable of accurately predicting nuclear properties across the nuclear chart. This approach enables the rapid evaluation of nuclear properties across multiple isotopes simultaneously. We benchmark our developments using the oxygen isotopic chain, achieving precise results for ground-state energies and nuclear charge radii, while providing robust uncertainty quantification. Our method demonstrates robust extrapolation and interpolation capabilities to nuclei outside of the training data, making it highly effective for transfer learning and enabling predictions in extreme regions of the nuclear chart. The emulator's flexible architecture lays the foundation for enabling global uncertainty quantification across the nuclear landscape, as well as guiding searches at rare isotope beam facilities towards isotopes that will reveal the most about the underlying nuclear forces.

### Code

This repository contains the code and data for reproducing the results in the paper 'Global Framework for Simultaneous Emulation Across the Nuclear Landscape'. The code is designed to train a Bayesian neural network to emulate the results of nuclear calculations many-body calculations.

## Installation

To install the code, clone the repository and install the required dependencies using pip:

```
git clone
cd bannane
pip install -r requirements.txt
```

## Structure

The repository is structured as follows:
```
paper_o_bannane/
│
├── data/                   # Data files
│   ├── all_o/             # Oxygen isotopic chain data used for training, evaluation, and testing
|
├── notebooks/
│   ├── simple/             # Training example of the model and benchmarking
│   ├── studies/            # Studies of the model's performance
|
├── LABS/                   # Data files output by the model for reproducibility of the studies
|
├── scripts/                 # Scripts for training and evaluating the model
│   ├── trainers             # Script for training the model
│   ├── slurm                # Script for training the model on a cluster for ablation studies

```


## License

This code is released under the MIT License.

## Authors

Please cite:
```bibtex

```

EMA-LAB @ MIT
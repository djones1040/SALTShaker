# SALTShaker

[![PyPI version](https://badge.fury.io/py/saltshaker-sn.svg)](https://badge.fury.io/py/saltshaker-sn)
[![Documentation Status](https://readthedocs.org/projects/saltshaker/badge/?version=latest)](https://saltshaker.readthedocs.io/en/latest/?badge=latest)
[![License: BSD](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

An open-source training framework for the SALT3 spectrophotometric model of Type Ia supernovae.

## Overview

SALTShaker trains the SALT3 (Spectral Adaptive Light-curve Template 3) model used for standardizing Type Ia supernovae in cosmological distance measurements. The framework:

- Trains spectrophotometric models from combined photometry and spectroscopy data
- In conjunction with SNANA provides a complete pipeline from simulation through cosmological fitting
- Supports custom survey configurations and filter systems

SALT3 is the successor to SALT2 ([Guy et al., 2007](https://ui.adsabs.harvard.edu/abs/2007A%26A...466...11G/abstract); [Guy et al., 2010](https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...7G/abstract)) and has been used in major cosmological analyses including Dovekie, DES5YR, and UNITY.

## Features

- **Model Training**: Train SALT3 models on your own photometric and spectroscopic datasets
- **Validation Suite**: Comprehensive tools to validate trained models against simulations
- **Full Pipeline**: End-to-end workflow from BYOSED simulations to cosmological parameter estimation
- **Flexible Configuration**: Extensive configuration options for regularization, priors, and optimization
- **Multi-Survey Support**: Built-in support for major surveys with K-correction files included
- **JAX Backend**: GPU-accelerated optimization using JAX

## Installation

### From PyPI (recommended)

```bash
pip install saltshaker-sn
```

### From Source

```bash
git clone https://github.com/djones1040/SALTShaker.git
cd SALTShaker
pip install -e .
```

### Dependencies

Core dependencies include: numpy, scipy, astropy, jax, sncosmo, emcee, iminuit, matplotlib, pandas

See `setup.py` for the complete list.

## Quick Start

### Download Example Training Data

```bash
trainsalt -g
```

This downloads the public SALT3 training dataset to `$OSTRICH_INITFILES`.

### Train a Model

```bash
trainsalt -c /path/to/training.conf
```

Or use the included example configuration:

```bash
cd examples/SALT3TRAIN_K21_PUBLIC
trainsalt -c SALT3_training.conf
```

### Use SALT3 for Light Curve Fitting

Once trained (or using the published model), fit supernova light curves with sncosmo:

```python
import sncosmo

data = sncosmo.load_example_data()
model = sncosmo.Model(source=sncosmo.SALT3Source(outputdir))
res, fitted_model = sncosmo.fit_lc(
    data, model,
    ['z', 't0', 'x0', 'x1', 'c'],
    bounds={'z': (0.3, 0.7)}
)
sncosmo.plot_lc(data, model=fitted_model, errors=res.errors)
```

## Documentation

Full documentation is available at **[saltshaker.readthedocs.io](https://saltshaker.readthedocs.io/en/latest/)**

- [Installation Guide](https://saltshaker.readthedocs.io/en/latest/install.html)
- [Getting Started](https://saltshaker.readthedocs.io/en/latest/gettingstarted.html)
- [Training Configuration](https://saltshaker.readthedocs.io/en/latest/training.html)
- [Data Formats](https://saltshaker.readthedocs.io/en/latest/data.html)
- [Pipeline Usage](https://saltshaker.readthedocs.io/en/latest/pipeline.html)

## Model Downloads

- **Latest SALT3 model**: Available in the [documentation](https://saltshaker.readthedocs.io/en/latest/)
- **Training data**: Included with `trainsalt -g` or from documentation

## Citation

If you use SALTShaker or the SALT3 model, please cite:

**Primary SALT3 paper:**
> Kenworthy et al., 2021, ApJ, 923, 265K
> [arXiv:2104.07795](https://arxiv.org/abs/2104.07795)

### Additional Publications

- [Pierel et al., 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...911...96P/abstract) - BYOSED simulation framework
- [Pierel et al., 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...939...11P/abstract) - Near-infrared SALT3 extension
- [Dai et al., 2023](https://ui.adsabs.harvard.edu/abs/2022arXiv221206879D/abstract) - Pipeline validation
- [Jones et al., 2023](https://ui.adsabs.harvard.edu/abs/2022arXiv220905584J/abstract) - Host-mass dependent model

## Contributing

We welcome contributions! Please:

1. Report bugs and request features via [GitHub Issues](https://github.com/djones1040/SALTShaker/issues)
2. Submit pull requests for bug fixes and enhancements
3. See the documentation for development setup

## License

This project is licensed under the BSD License - see the LICENSE file for details.

## Authors

- David Jones (dojones@hawaii.edu)
- D'Arcy Kenworthy
- Rick Kessler
- Mi Dai
- Justin Pierel

## Acknowledgments

SALTShaker builds on the original SALT/SALT2 training code developed by J. Guy. Development has been supported by NASA, DOE, and the Gordon and Betty Moore Foundation.

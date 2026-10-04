# DASMF for Coal Body Structure Classification

PyTorch implementation of DASMF for classifying coal body structures from electrical microresistivity imaging (ERMI) images.

## Overview

DASMF uses a ResNet-18 backbone with two components:

- **Directional adaptive scale routing (DASR)** combines depth-directed and azimuth-directed feature responses at multiple scales.
- **Multi-level feature fusion (MLF)** combines features from three backbone stages for four-class classification.

The model accepts single-channel grayscale images and outputs four class logits.

### Coal structure categories

`0_primary`, `1_cataclastic`, `2_granulated`, `3_mylonitic`

## Repository contents

- `dasmf.py`: DASMF architecture and selectable loss functions.
- `train_dasmf.py`: training and evaluation script.
- `make_demo_dataset.py`: generator for the synthetic demonstration dataset.
- `demo_synthetic_dataset/`: synthetic images organized into training, validation, and test folders.
- `requirements.txt`: Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Data availability

The field ERMI images and associated core data are subject to geological data confidentiality restrictions and cannot be publicly released. The repository therefore provides a **synthetic demonstration dataset** for checking the code and data-loading workflow.

Synthetic images do not represent real coal body structures. Results obtained from them must not be interpreted as geological findings or as a reproduction of the performance reported for the independent test well.

The dataset uses the following directory structure:

```text
demo_synthetic_dataset/
├── train/
│   ├── 0_primary/
│   ├── 1_cataclastic/
│   ├── 2_granulated/
│   └── 3_mylonitic/
├── val/
│   ├── 0_primary/
│   ├── 1_cataclastic/
│   ├── 2_granulated/
│   └── 3_mylonitic/
└── test/
    ├── 0_primary/
    ├── 1_cataclastic/
    ├── 2_granulated/
    └── 3_mylonitic/
```

The demonstration images can be generated with:

```bash
python make_demo_dataset.py
```

## Model and training

`dasmf.py` defines `DASMFResNet18`, `DepthAzimuthScaleRouter`, and `MultiLevelClassifier`. It also provides cross-entropy, focal loss, and focal loss with a categorical-index penalty for the loss-function comparison.

**Cross-entropy was used for the final DASMF architecture and model comparisons.** Defining the other losses in the code does not mean that they were used to train the final reported model.

Run the demonstration training workflow with:

```bash
python train_dasmf.py
```

This command uses the synthetic dataset to check that the training and evaluation pipeline runs. The reported field-data results require the confidential, core-calibrated ERMI dataset and its well-level partition; they cannot be reproduced from the synthetic images.

## Citation

If you use this code, please cite the associated paper when its bibliographic details become available.

## License

This project is distributed under the MIT License. See `LICENSE`.

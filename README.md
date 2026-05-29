# Conformal Event Prediction with Temporal Knowledge Graph

This repository contains the implementation of **CFEP**, proposed in the paper
**"Conformal Event Prediction with Temporal Knowledge Graph"**. The paper has
been accepted by **ACL 2026**.

## Overview

CFEP is a conformal prediction framework for event prediction on temporal
knowledge graphs. Instead of producing only point predictions, CFEP constructs
prediction sets with statistical coverage guarantees, making event prediction
more reliable in high-stakes scenarios.

The method addresses the non-exchangeability issue in temporal knowledge graph
event prediction and improves the efficiency of prediction sets while preserving
the target coverage.

## Method

CFEP contains two main modules:

- **Non-conformity score diffusion**: diffuses non-conformity scores through
  temporal and topological neighbors in the temporal knowledge graph, so that
  related events obtain more stable uncertainty estimates.
- **Efficiency-aware optimization**: learns weighted quantiles to reduce the
  coverage gap and produce more compact prediction sets while maintaining the
  required coverage level.

The overall pipeline first trains an event prediction backbone, then computes
base non-conformity scores on the calibration set, diffuses these scores using
temporal/topological event relations, and finally constructs prediction sets on
the test set.

## Repository Structure

```text
CFEP/
|-- README.md
|-- train_cp.py                 # main training and conformal evaluation script
|-- models.py                   # backbone event prediction models
|-- propagations.py             # graph propagation layers
|-- modules_f.py                # attention and neural utility modules
|-- event_data_processing.py    # temporal event preprocessing utilities
|-- event_data_processing_ml.py # multi-label distribution utilities
|-- event_type.csv              # event type metadata
`-- data/
    `-- EG/
        |-- stat.txt
        |-- train.txt
        |-- valid.txt
        `-- test.txt
```

## Data Preparation

The original data are event records from three regions: Egypt (EG), Iran (IR),
and Israel (IS). Each event is represented as a temporal knowledge graph edge
with the format:

```text
head relation tail time
```

The preprocessing pipeline follows these steps:

1. Convert actor names and event codes into integer IDs.
2. Split data into training, validation, calibration-training,
   calibration-validation, and test sets.
3. Encode news text with a pretrained language model, such as
   `bge_base_en_v1.5`.
4. Build time-indexed directed event graphs with text embeddings.

## Usage

The main entry point is:

```bash
python train_cp.py --model glean --dataset <DATASET_NAME> --dp <DATA_ROOT>/
```

For conformal prediction experiments, the expected dataset directory contains:

```text
train.txt
valid.txt
calib_train.txt
calib_valid.txt
test.txt
stat.txt
```

During training, the backbone event prediction model is trained on the training
set, the conformal model is optimized on the calibration-training set, the
quantile threshold is estimated on the calibration-validation set, and coverage
and efficiency are evaluated on the test set.

## Evaluation

CFEP is evaluated with two metrics:

- **Coverage**: the fraction of test instances whose true event type is included
  in the prediction set.
- **Efficiency**: the average size of the prediction set; smaller sets indicate
  higher efficiency when coverage is preserved.

## Notes

This repository currently provides the core source code and an example processed
EG split. Some generated artifacts, full raw data, preprocessed graph dictionaries,
and dependency files may need to be prepared separately for full reproduction.

# Zero-Shot Brain Stimuli Classification

A compact research workflow for classifying **unseen EEG classes** by mapping brain signals into shared **text** and **image** embedding spaces.

The project is currently notebook-driven (`DataAndModel.ipynb`) and uses the ThingsEEG-Text dataset with multimodal prototype matching.

## Repository Contents

- `DataAndModel.ipynb` — full analysis + modeling pipeline (data loading, preprocessing, baselines, main model, evaluation)
- `report.pdf` — accompanying report document
- `README.md` — project documentation

## Problem Setting

Given EEG recordings from seen classes, learn mappings from EEG features to:
- CLIP text embeddings
- visual embeddings (Cornet-S PCA features)

Then perform **zero-shot classification** on unseen EEG samples by:
1. Predicting text/image embeddings from EEG
2. Comparing predictions with class prototypes in embedding space
3. Fusing modalities with weighted cosine similarity

## Dataset Layout Expected by the Notebook

The notebook expects a local dataset root like:

`ThingsEEG-Text/`

With subfolders used in code:
- `brain_feature/17channels/sub-XX/`
- `textual_feature/ThingsTrain/text/CLIPText/sub-XX/`
- `textual_feature/ThingsTest/text/CLIPText/sub-XX/`
- `visual_feature/ThingsTrain/pytorch/cornet_s/sub-XX/`
- `visual_feature/ThingsTest/pytorch/cornet_s/sub-XX/`

Key files loaded include:
- `eeg_train_data_within.mat`
- `eeg_test_data.mat`
- `eeg_test_data_unique.mat`
- `text_feat_train.mat`, `text_feat_test.mat`, `text_feat_test_unique.mat`
- `feat_pca_train.mat`, `feat_pca_test.mat`, `feat_pca_test_unique.mat`

## Data Shapes (from notebook outputs)

For a single subject example:
- Seen EEG: `(16540, 17, 100)`
- Unseen EEG: `(16000, 17, 100)`
- Unique EEG: `(200, 17, 100)`
- Unseen text embeddings: `(16000, 512)`
- Unseen image embeddings: `(16000, 1000)`

## Method Overview

The notebook compares multiple variants:

1. **Baseline fused prototype matching**
   - Flatten EEG
   - Ridge regression to text/image embeddings
   - Classify with cosine prototype matching

2. **Main model (recommended notebook section)**
   - Rolling z-score normalization on EEG
   - Class-wise split on seen classes for alpha validation
   - Ridge alpha search
   - Retrain on all seen classes
   - Evaluate top-1 and top-5 on unseen classes

3. **Ablation variants**
   - rolling-zscore only
   - PCA drop experiment

## Reported Results (current notebook run)

- Baseline fused unseen accuracy: **0.0741**
- Rolling-zscore + fixed-alpha fused unseen accuracy: **0.1006**
- Main model final unseen top-1 accuracy: **0.1115 (11.15%)**
- Main model final unseen top-5 accuracy: **0.2919 (29.19%)**

## Environment

Notebook imports indicate the following Python packages are required:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `IPython`

## Running the Project

1. Prepare the ThingsEEG-Text dataset in the expected folder structure.
2. Open `DataAndModel.ipynb` in Jupyter.
3. Update dataset root/subject variables in the notebook if needed.
4. Run cells in order from top to bottom.

## Current Limitations / Next Work

The notebook TODO notes an open step:
- compute an improved strategy for selecting the optimal fusion weight (`alpha`) between text and image mappings.

## Notes

- Paths in the notebook currently use Windows-style separators in several template strings.
- This repository is centered on reproducible experimentation in a single notebook rather than a packaged training pipeline.

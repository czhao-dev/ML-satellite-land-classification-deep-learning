# Satellite Land Classification with CNNs and Vision Transformers

This portfolio project classifies satellite image tiles as **agricultural** or **non-agricultural** land using deep learning models built in both Keras/TensorFlow and PyTorch.

The work progresses from data-loading experiments to CNN baselines, then to CNN-Vision Transformer hybrid models that combine local feature extraction with global attention.

## Project Highlights

- Binary land-use classification from satellite image tiles
- 6,000 image dataset with balanced classes
- Keras and PyTorch data pipelines
- CNN baselines in both frameworks
- CNN-ViT hybrid models in both frameworks
- Cross-framework evaluation using accuracy, precision, recall, F1, ROC-AUC, loss, and confusion matrices

## Dataset

The dataset contains 6,000 JPG satellite tiles:

| Class | Meaning | Images |
|---|---|---:|
| `class_0_non_agri` | Non-agricultural land | 3,000 |
| `class_1_agri` | Agricultural land | 3,000 |

The original lab notebooks download the dataset from IBM Skills Network cloud storage:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4Z1fwRR295-1O3PMQBH6Dg/images-dataSAT.tar
```

Large local data files are kept out of Git. See `data/README.md` for setup notes.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Keras CNN | 0.9925 | 1.0000 | 0.9850 | 0.9924 | 1.0000 |
| PyTorch CNN | 0.9988 | 0.9983 | 0.9993 | 0.9988 | 1.0000 |
| Keras CNN-ViT Hybrid | 0.9958 | 0.9990 | 0.9927 | 0.9958 | 0.9998 |
| PyTorch CNN-ViT Hybrid | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 1.0000 |

The PyTorch models achieved the strongest scores in these runs, with the final PyTorch CNN-ViT hybrid reaching 99.90% accuracy.

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   └── raw/                  # Local dataset archive, ignored by Git
├── models/
│   ├── README.md
│   └── trained/              # Local model artifacts, ignored by Git
├── reports/
│   └── results_summary.md
├── scripts/
│   ├── 01_data_loading_memory_vs_generator.py
│   ├── 02_keras_data_pipeline.py
│   ├── 03_pytorch_data_pipeline.py
│   ├── 04_keras_cnn_classifier.py
│   ├── 05_pytorch_cnn_classifier.py
│   ├── 06_keras_vs_pytorch_cnn_comparison.py
│   ├── 07_keras_cnn_vit_hybrid.py
│   ├── 08_pytorch_cnn_vit_hybrid.py
│   └── 09_final_cnn_vit_evaluation.py
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── visualization.py
├── LICENSE
└── requirements.txt
```

## Notebook Guide

Run the notebooks in order if you want to reproduce the full workflow:

1. Compare memory-based and generator-based image loading.
2. Build Keras data loading and augmentation pipelines.
3. Build PyTorch data loading and augmentation pipelines.
4. Train and evaluate a Keras CNN classifier.
5. Train and evaluate a PyTorch CNN classifier.
6. Compare Keras and PyTorch CNN models.
7. Build a Keras CNN-ViT hybrid model.
8. Build a PyTorch CNN-ViT hybrid model.
9. Evaluate and compare the final CNN-ViT hybrid models.

## Python Scripts

The `scripts/` folder contains cleaned `.py` exports of the notebooks. These are included so GitHub reviewers can browse the project as source code without opening every notebook.

The exported scripts are:

1. `01_data_loading_memory_vs_generator.py`
2. `02_keras_data_pipeline.py`
3. `03_pytorch_data_pipeline.py`
4. `04_keras_cnn_classifier.py`
5. `05_pytorch_cnn_classifier.py`
6. `06_keras_vs_pytorch_cnn_comparison.py`
7. `07_keras_cnn_vit_hybrid.py`
8. `08_pytorch_cnn_vit_hybrid.py`
9. `09_final_cnn_vit_evaluation.py`

The notebooks remain the best narrative version because they preserve explanations, outputs, and visualizations. The scripts are better for quick code review, search, and future refactoring.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

## Portfolio Notes

This project began as an IBM/Coursera deep learning capstone lab sequence. I reorganized it into a portfolio-ready repository with a clear workflow, documented results, reusable helper modules, and GitHub-friendly handling for large data and model artifacts.

## Next Improvements

- Add a small inference script for classifying a new satellite tile.
- Export selected plots from notebooks into `reports/figures/`.
- Add a lightweight Streamlit demo for interactive predictions.
- Track experiments with a reproducible configuration file.

## License

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full license text.


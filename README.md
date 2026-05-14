# Deep Learning for Automated Sickle Cell Detection in Peripheral Blood Smears

An explainable deep learning project for automated sickle cell disease detection using African clinical microscopy data, built as part of MSB7216: Deep Learning for Health Data at Makerere University.

**Author:** Okidi Patrovas Gaabriel | 2025/HD07/26020U
**Institution:** Makerere University, Kampala, Uganda
**Course:** MSB7216: Deep Learning for Health Data

## Live Demo

Launch the Screening Tool: https://huggingface.co/spaces/Patrovas/sickle-cell-detection

Upload a blood smear image and get an instant prediction with Grad-CAM explainability showing exactly which regions of the image the model focused on.

## Project Overview

Sickle cell disease affects an estimated 13.3% of the Ugandan population, with prevalence reaching 19.8% in Kampala alone. Diagnosis currently requires trained laboratory personnel, specialist equipment, and manual microscopic examination — resources that are largely inaccessible in rural and low-resource settings across Uganda. This project builds an explainable deep learning model for automated sickle cell detection from blood smear images, designed specifically for low-resource, mobile-first point-of-care screening. This is the first deep learning study to apply transfer learning and explainability to the Tushabe (2024) Ugandan clinical mobile-phone microscopy dataset.

## Datasets

**Primary Dataset — Positive Class:** Tushabe et al. (2024) Ugandan Sickle Cell Microscopy Dataset. 422 unlabelled blood smear images collected from patients in Soroti and Kumi districts, Uganda, captured using mobile phone cameras placed on basic microscopes under real clinical conditions. Published in Acta Scientific Microbiology, Volume 7, Issue 12. Available at https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset

**Supplementary Dataset — Negative Class:** BCCD Dataset (Shenggan et al.). 364 normal peripheral blood smear images used to supplement the negative class. MIT License. Available at https://github.com/Shenggan/BCCD_Dataset

**Additional Negative Samples:** 147 normal blood smear images from the Tushabe (2024) dataset.

Combined raw total: 933 images across both classes. Split: 653 training / 140 validation / 140 test using a stratified 70/15/15 ratio.

## Results

| Model | Accuracy | AUC-ROC | Sensitivity | Specificity | F1 Score |
|---|---|---|---|---|---|
| Baseline CNN (from scratch) | 87.86% | 0.9588 | 95.24% | 81.82% | 0.8759 |
| EfficientNet-B0 (transfer learning) | 96.43% | 0.9951 | 95.24% | 97.40% | 0.9600 |
| ResNet-50 (transfer learning) | 98.57% | 0.9973 | 98.41% | 98.70% | 0.9841 |

The best performing model is ResNet-50 with unfrozen transfer learning, achieving only 2 errors on 140 test images — 1 false positive and 1 false negative. Grad-CAM explainability confirms the model attends to biologically meaningful cell regions rather than imaging artefacts.

## Project Structure

The repository is organised as follows. The data folder contains raw and processed image datasets. The notebooks folder contains six Jupyter notebooks covering EDA, preprocessing, baseline modelling, EfficientNet training, ResNet training, and Grad-CAM explainability. The src folder contains Python source code for the dataset loader, model definitions, training loop, and evaluation utilities. The models folder stores saved model weights. The reports folder contains the final project report. The figures folder contains all EDA plots and Grad-CAM visualisations generated during the project. The app folder contains the Gradio deployment application.

## Setup Instructions

Clone this repository using git clone https://github.com/Patro331/sickle-cell-detection.git and navigate into the project folder. Install all dependencies by running pip install torch torchvision gradio huggingface_hub opencv-python matplotlib scikit-learn seaborn Pillow numpy. Download the datasets from the links above and place sickle cell images in data/raw/positive/ and normal images in data/raw/negative/. Run the notebooks in order from 01 through 06. To launch the Gradio demo locally, run python app/app.py from the project root.

## Notebooks

01_eda.ipynb covers exploratory data analysis including class distribution, image size analysis, pixel intensity analysis, and image quality checking across all 933 images. 02_preprocessing.ipynb implements the full preprocessing pipeline including resizing to 224x224, ImageNet normalisation, data augmentation, and stratified train/validation/test splitting. 03_baseline_model.ipynb trains a simple 3-block CNN from scratch to establish a performance baseline of 87.86% accuracy. 04_efficientnet.ipynb fine-tunes EfficientNet-B0 with frozen and unfrozen backbone strategies, achieving 96.43% accuracy with the unfrozen approach. 05_resnet.ipynb fine-tunes ResNet-50 with frozen and unfrozen backbone strategies, achieving the best result of 98.57% accuracy. 06_explainability_gradcam.ipynb applies Grad-CAM to ResNet-50 across all four prediction categories — true positives, true negatives, false positives, and false negatives — with detailed error analysis.

## Key Findings

Transfer learning with ResNet-50 achieves 98.57% accuracy using only 653 training images, demonstrating that pretrained ImageNet features transfer effectively to blood smear microscopy. Grad-CAM confirms the model attends to biologically meaningful regions within the circular microscope field rather than imaging artefacts or the black border. The single false negative had a confidence score of 0.472, just below the 0.5 decision threshold — lowering the threshold to 0.4 would eliminate this error at the cost of one additional false positive. The single false positive was triggered by a staining artefact, highlighting artefact detection as a key area for future preprocessing improvement. EfficientNet-B0 trains more stably than ResNet-50 on this small dataset due to its more parameter-efficient architecture, though ResNet-50 ultimately achieves superior final test performance.

## Ethical Considerations

This model is a screening tool only and does not replace diagnosis by a trained clinician. All positive predictions must be confirmed by a qualified medical professional before any clinical action is taken. Both datasets were collected with documented patient consent and are freely available for research and educational use. The domain gap between Tushabe mobile-phone microscopy images and BCCD laboratory images represents a known limitation that future work should address by sourcing additional normal smear images captured under mobile-phone microscope conditions consistent with the Tushabe dataset.

## Citation

Tushabe, F. et al. (2024). A Dataset of Microscopic Images of Sickle and Normal Red Blood Cells. Acta Scientific Microbiology, 7(12). Available at https://actascientific.com/ASMI/pdf/ASMI-07-1453.pdf

Acevedo, A. et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Computer Methods and Programs in Biomedicine. DOI: 10.1016/j.cmpb.2019.105020

Shenggan et al. BCCD Dataset. Available at https://github.com/Shenggan/BCCD_Dataset

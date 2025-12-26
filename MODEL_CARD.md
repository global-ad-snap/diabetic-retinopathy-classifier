# Model Card — Diabetic Retinopathy Classifier

## Model Name
Diabetic Retinopathy Image Classification CNN

## Model Purpose
To classify retinal fundus images into diabetic retinopathy severity stages for research-oriented decision-support prototyping and model interpretability demonstration.

## Training Data
- Dataset: APTOS 2019 Blindness Detection Dataset
- Data Type: Retinal fundus images
- Classes: No DR, Mild, Moderate, Severe, Proliferative DR
- Data Source: Public Kaggle dataset
- Limitations: Retrospective data, class imbalance, variable image quality

## Model Architecture
- Algorithm: Custom Convolutional Neural Network (CNN)
- Input Resolution: 128 × 128
- Framework: PyTorch
- Training Approach: Supervised multi-class classification

## Evaluation Metrics
- Metrics: ROC-AUC, confusion matrix, accuracy
- Validation Method: Train–validation split

Performance metrics are reported on the available validation split and may not reflect real-world clinical performance.

## Performance Summary
The model demonstrated reasonable discriminatory performance across diabetic retinopathy severity levels on the validation dataset. Performance varied across classes, reflecting dataset imbalance and visual similarity between adjacent severity stages.

## Interpretability
- Grad-CAM: Visualizes class-specific activation regions
- SHAP: Provides pixel-level feature attribution
These methods are applied post-hoc and reflect model behavior rather than causal clinical reasoning.

## Limitations
- No external or prospective validation
- Dataset-specific bias
- Not evaluated across multiple imaging devices or populations

## Ethical Considerations
- Risk of misuse without clinical oversight
- Potential bias due to dataset composition
- Interpretability outputs may be over-interpreted without domain expertise

## Intended Use Statement
These models are intended for research and educational purposes, including portfolio demonstration, and are not approved for clinical diagnosis, treatment decisions, or operational deployment without further validation.

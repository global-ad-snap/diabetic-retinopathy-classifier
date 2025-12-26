# Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset, which contains retinal fundus images labeled by diabetic retinopathy severity.

Due to **dataset licensing and medical data considerations**, the dataset is **not included** in this repository.

---

## Dataset Source

APTOS 2019 Blindness Detection (Kaggle)  
https://www.kaggle.com/c/aptos2019-blindness-detection

---

## Expected Directory Structure

After downloading and preparing the dataset, the `data/` directory should be structured as follows:

```text
data/
├── train_images/      # Training retinal images
├── test_images/       # Test retinal images
├── train.csv          # Training labels and metadata
├── test.csv           # Test metadata
└── valid.csv          # Validation split metadata
```
---

## Notes

- Images are expected to be RGB fundus photographs.
- CSV files should follow the original APTOS 2019 schema.
- Preprocessing (resizing, normalization, augmentation) is handled in
  `src/data_utils.py`.
- The dataset should be placed exactly in this structure for the training
  and evaluation scripts to work correctly.

---

## Disclaimer

The dataset consists of medical images and should be handled responsibly.
This project is intended for **educational and research purposes only** and
does not constitute a medical device or clinical diagnostic tool.

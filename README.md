
# A Comparative Study of Transformer-Based Models and Statistical Models for Multi-Class Disease Classification

This repository contains the source code, experimental notebooks, and documentation for the M.Tech project titled **" A Comparative Study of Transformer-Based Models and Statistical Models for Multi-Class Disease Classification"**, conducted at the Indian Institute of Technology Madras, under the guidance of **Prof. Balaraman Ravindran** and **Prof. Gokul S Krishnan**.

## Project Overview

The project addresses the challenge of classifying biomedical abstracts into multiple disease categories using advanced Natural Language Processing (NLP) models. Four transformer-based models**BioBERT**, **BERT**, **XLNet**, and a custom lightweight model named **LastBERT**are evaluated for their effectiveness in classifying abstracts into:

- Neoplasms
- Digestive System Diseases
- Nervous System Diseases
- Cardiovascular Diseases

These are benchmarked against traditional models such as **SVM**, **Random Forest**, and **Logistic Regression**.

## Key Highlights

- **BioBERT** achieved the highest accuracy of **97%**, followed by **XLNet (96%)**, **BERT (87%)**, and **LastBERT (85%)**.
- **LastBERT**, with only **~29M parameters**, offers an optimal trade-off between accuracy and computational efficiency.
- Evaluation metrics include: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Confusion Matrix**, and **ROC-AUC**.
- Designed for real-world clinical use, especially in **resource-constrained environments**.

##  Repository Structure

```
 biobert-final.ipynb        # Fine-tuning and evaluation of BioBERT
 bert4last-final.ipynb      # Fine-tuning of BERT for comparison with LastBERT
 lastbert-final.ipynb       # Lightweight custom transformer implementation
 xlnet-final.ipynb          # Fine-tuning and evaluation of XLNet
 GE23M018_Report_MTech.pdf  # Full thesis report (converted if needed)
 README.md                  # This file
```

## Technologies Used

- **Python 3.10**
- **PyTorch 2.0+**
- **Hugging Face Transformers 4.39+**
- **Google Colab / Kaggle GPU (Tesla T4/P100)**
- Libraries: `datasets`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`

## Dataset

- **Medical-Abstracts-TC-Corpus**
- 14,438 biomedical abstracts categorized into 5 disease classes
- After filtering, 4 major disease categories were selected
- Upsampled to address class imbalance

## How to Run


1. Open any of the notebooks (`.ipynb`) in [Google Colab](https://colab.research.google.com/) or Jupyter.

2. Make sure to install dependencies:
   ```bash
   pip install transformers datasets scikit-learn pandas matplotlib seaborn
   ```

3. Run the notebooks to reproduce the results.

## Results Summary

| Model     | Accuracy | ROC-AUC | Params        | Notes                         |
|-----------|----------|---------|---------------|-------------------------------|
| BioBERT   | 97%      | ~1.00   | 110M          | Best performance overall      |
| XLNet     | 96%      | ~1.00   | 110M+         | Robust without domain tuning  |
| BERT      | 87%      | ~0.93   | 110M          | General-purpose baseline      |
| LastBERT  | 85%      | ~0.92   | ~29M          | Lightweight and efficient     |
| SVM       | ~57%     | -       | -             | Traditional baseline          |
| LR        | ~45%     | -       | -             | Traditional baseline          |   
 

## Contributions

- Comprehensive benchmarking of transformer models vs classical ML models.
- Designed **LastBERT**, a lightweight transformer optimized for deployment.
- Developed a unified pipeline for multi-class classification.

## Author

**Abdul Ahad Ansari**
Roll No.: GE23M018  
M.Tech Data Science and Artificial Intelligence  
Indian Institute of Technology Madras  
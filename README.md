## 🧠 Project Summary

Epilepsy affects over 50 million people worldwide and is commonly diagnosed through the analysis of EEG (electroencephalogram) signals. 
However, EEG data is highly complex and often difficult to interpret, even for skilled medical professionals. 
While machine learning (ML) and neural network methods can achieve high prediction accuracy, they are sometimes viewed with skepticism due to their lack of transparency and explainability in clinical contexts.

This project takes a balanced approach that prioritizes both **model performance and interpretability**. The goal is to build reliable seizure detection models from EEG data while providing meaningful insights into how the models make decisions.

### ✅ Key Highlights:

- **Dataset**: Epileptic Seizure Recognition dataset from [Kaggle](https://www.kaggle.com/)
- **Data Preparation**:
  - Feature Reduction
  - Feature Scaling
  - Conversion to binary classification (seizure vs non-seizure)

- **Models Evaluated**:
  - Logistic Regression  
  - Decision Tree  
  - Naive Bayes  
  - AdaBoost (over Decision Tree and Naive Bayes)  
  - 2-layer Backpropagation Neural Network

- **Performance**:
  - Best accuracy of **96.33%** achieved by **AdaBoost over Decision Tree**
  - Feature scaling impact observed to vary by model

- **Explainability**:
  - Simple, interpretable techniques applied to Logistic Regression, Decision Tree, and Naive Bayes
  - Visual interpretation plots created for better understanding
  - **Naive Bayes** identified as the best trade-off between accuracy and interpretability

- **Conclusion**:
  - Highlights the need for **explainable AI (XAI)** in healthcare
  - Emphasizes an intuitive, data-driven approach to seizure detection using EEG

## 📂 Notebooks & Scripts

| File Name                          | Description                         | 
|-----------------------------------|--------------------------------------|
| `EEG_1_EDA.ipynb`                 | Exploratory Data Analysis            |
| `EEG_2_ALLmodels.ipynb`           | Evaluation of multiple ML models     | 
| `EEG_3_ResearchModels (1).ipynb`  | Final selected models for research   | 
| `EEG_eegdataloder_doc (1).ipynb`  | Explanation for data loading
|                                     functions eeg_dataloader.py 
| `InterpretabilityPlots (1).ipynb` | Visualization of model explanations  | 
| `eeg_dataloader.py`               | Custom EEG data loader module        | 


### AI-ThoracicSurvivalPrediction

This project is a big data analysis and machine learning-based predictive modeling project focused on thoracic surgery patients. The goal is to predict 1-year survival after surgery using clinical pre-operative data.

---

### Project Overview

This project utilizes clinical data from thoracic surgery patients to analyze correlations between pre-operative features and survival outcomes. Using data-driven methods, we built multiple classification models to predict whether a patient will survive within 1 year after surgery.

---

### Objectives

* Identify significant clinical variables associated with post-surgical survival.
* Develop predictive models using big data techniques.
* Evaluate model performance and provide insights for personalized treatment planning.

---

### Dataset

* Source: UCI Machine Learning Repository
  [Thoracic Surgery Data](https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data)
* Total instances: 470
* Features: 17 input variables (DGN, PRE4-PRE32, AGE), 1 output variable (Risk1Yr: 1-year survival indicator)

---

### Data Preprocessing

* **Normalization**: Min-Max normalization for PRE4, PRE5
* **Label Encoding**: Applied to categorical variables (e.g., PRE6, PRE14)
* **Binary Encoding**: T/F values (PRE7\~PRE32, Risk1Yr) converted to 0/1
* **Age Binning**: AGE values grouped into 10-year bins
* **Missing/Outlier Handling**: No missing values; outliers included with guidance

---

### Data Analysis & Visualization

* Distribution plots of diagnostic codes (DGN), tumor size (PRE14), and smoking status
* Histograms of continuous variables (PRE4, PRE5)
* Visualizations of binary features like coughing, pain, etc.
* Correlation analysis using GLM
* Imbalanced data handling with ROSE for the target variable (Risk1Yr)

---

### Hypothesis

* Null hypothesis: No independent variable significantly affects survival
* Alternative hypothesis: At least one variable significantly affects survival
  (Tested at significance level of 0.05)

---

### Modeling Techniques

1. **Random Forest**

   * Accuracy: 95%
2. **XGBoost**

   * Accuracy: 96%
3. **SVM**

   * Initial accuracy: 76%
   * Tuned accuracy: 96%
4. **Logistic Regression**

   * Accuracy: \~67% (improved slightly using ROC-based threshold tuning)

---

### Key Findings

* **Most important variable**: PRE14 (Primary tumor size)
* **Secondary variable**: PRE17 (Diabetes status)
* Other relevant features: PRE4, PRE5 (pulmonary function), DGN codes (e.g., DGN3, DGN5), and age ranges

---

### Conclusion

* Predictive models can assist in identifying high-risk patients pre-operatively
* Personalized treatment plans can be designed based on predictive features
* Big data techniques enable data-driven medical decision support

This project demonstrates the feasibility of using machine learning in thoracic surgery survival prediction, offering practical insights for real-world clinical applications.

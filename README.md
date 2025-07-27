# Titanic - Machine Learning from Disaster 🚢

This project is a solution to the classic Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). The goal is to build a predictive model that determines whether a passenger survived the Titanic shipwreck based on features such as age, class, sex, and more.

---

## Problem Statement

Given passenger data, build a model to predict whether they survived or not.

---

## Dataset

The dataset contains information such as:
- PassengerId
- Pclass (Ticket class)
- Name
- Sex
- Age
- SibSp (siblings/spouses aboard)
- Parch (parents/children aboard)
- Ticket
- Fare
- Cabin
- Embarked (port of embarkation)

You can find the dataset on Kaggle:  
🔗 [Titanic Dataset on Kaggle](https://www.kaggle.com/competitions/titanic/data)

---

## Tools & Libraries Used

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- Jupyter Notebook

---

## Approach

1. **Data Cleaning**  
   - Handle missing values (Age, Cabin, Embarked)
   - Drop unnecessary features (Ticket, Name, etc.)

2. **Feature Engineering**  
   - Convert categorical variables to numerical
   - Create new features

3. **Model Building**  
   - Trained classifiers like Logistic Regression, Decision Tree, Random Forest, etc.

4. **Model Evaluation**  
   - Used accuracy, confusion matrix, and cross-validation.

---
## Data Preprocessing & Modeling
This project includes several steps to prepare the dataset and apply different machine learning models for the Titanic survival prediction task.


**Handling Missing Values**

Missing entries in the "Cabin" column were filled with "0" to indicate the absence of cabin information.

Rows where the "Cabin" field did not have exactly 2 or 3 characters were removed to maintain consistency.

**Feature Engineering**

The "Cabin" field was encoded into numerical format for modeling purposes.

A new feature called ticket_type was created by extracting the prefix of the "Ticket" column.

Categorical variables such as "Sex" and "Embarked" were converted to numerical representations using label encoding or one-hot encoding as needed.

## Exploratory Data Analysis (EDA)
Basic exploratory analysis was performed to understand the distribution and significance of various features:

Age Distribution: Visualized with histograms to understand the survival trend across different age groups.

Ticket Type Counts: Analyzed the frequency of different ticket prefixes and their correlation with survival.

Gender Counts: Investigated the male-to-female ratio and its impact on survival outcomes.

## Model Selection

**Random Forest Classifier**
  - Pros: Handles both numerical and categorical data well, reduces variance through ensembling.

  - Cons: Can overfit if the trees are too deep or the model is not properly tuned.

  - Result: Moderate accuracy  (83%)

**Support Vector Machine (SVM)**
  - Pros: Captures complex non-linear relationships using kernels like RBF.

  - Cons: Sensitive to parameter tuning (C, gamma), can overfit with high C values and is slower on large datasets.

  - Result: Moderate accuracy (88%)

**Decision Tree Classifier (Best Performing)**
  - Pros: Easy to interpret, fast to train, and revealed key survival-related features clearly.

  - Cons: Prone to overfitting without pruning (though handled well in this case).

  - Result: Outperformed other models on validation accuracy and generalization.
Final Accuracy: 91%


## Results

Best accuracy score on validation set: 91%

---
## Authors

## License


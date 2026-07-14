## KNN and Data Science Mini Project

A mini data science project demonstrating machine learning workflows
from data preprocessing and exploratory analysis through clustering,
regression, classification, model evaluation, and interpretation using
student video-watching behavior from an online learning platform.

## The repository implements and evaluates:


## Data preprocessing and feature engineering
- Data cleaning and removal of non-informative attributes
- Missing-value handling
- Feature scaling and normalization
- Dataset transformation for machine learning pipelines
- Feature grouping for behavioral analysis


## Exploratory data analysis
- Statistical analysis of student video-watching behavior
- Behavioral feature categorization
- Visualization of relationships between viewing patterns and quiz performance
- Cluster interpretation based on engagement metrics


## Unsupervised learning
- K-Means clustering of student behavior
- Natural grouping of students based on video engagement
- Cluster labeling and behavioral interpretation
- Comparison of engagement patterns across groups


## Predictive modeling
- Linear regression for quiz performance prediction
- Training and testing data partitioning
- Mean Squared Error (MSE) evaluation
- Analysis of regression limitations for behavioral prediction


## Classification with K-Nearest Neighbors
- k-Nearest Neighbors (kNN) classification
- Student performance prediction from behavioral features
- Confusion matrix analysis
- Precision, recall, F1-score, and accuracy evaluation
- Comparison across multiple feature pair combinations


## Repository structure

knn-data-science-mini-project/
├── data/
│   └── behavior_performance.txt
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── regression.py
│   ├── classification.py
│   └── visualization.py
├── outputs/
│   ├── cluster_results.png
│   ├── regression_results.png
│   ├── confusion_matrices.png
│   └── feature_scatterplots.png
├── requirements.txt
└── README.md

The src/ directory contains reusable implementations for preprocessing,
clustering, regression, and classification. The analysis notebooks and
visualizations demonstrate the complete machine learning workflow from
raw data to model evaluation.

## Run the complete project
python -m venv .venv

Activate the environment and run:

pip install -r requirements.txt
python main.py

Or execute the Jupyter notebook for the complete analysis.

## Generated results

- cluster_results.png — K-Means clustering of student engagement behavior
- regression_results.png — Linear regression predictions and error analysis
- confusion_matrices.png — kNN classification performance
- feature_scatterplots.png — Behavioral feature relationships and class separation
- classification_report.txt — Precision, recall, F1-score, and accuracy metrics


## Engineering conclusions

Student video-watching behavior can be grouped into meaningful behavioral clusters using K-Means clustering.
Linear regression provides limited predictive power for quiz performance, suggesting that the relationship between engagement and learning outcomes is nonlinear.
k-Nearest Neighbors achieves substantially better predictive performance than linear regression for classifying student quiz outcomes.
Behavioral features such as pauses, playback rate, rewinds, and completion percentage contain strong information about student performance.
Appropriate preprocessing, scaling, and feature engineering significantly improve machine learning model performance.
Comparing supervised and unsupervised learning methods provides complementary insights into student learning behavior.

## Skills

Machine learning, K-Nearest Neighbors (kNN), K-Means clustering, linear regression, supervised learning, unsupervised learning, feature engineering, data preprocessing, classification metrics, confusion matrices, precision and recall analysis, Mean Squared Error (MSE), exploratory data analysis (EDA), NumPy, Pandas, scikit-learn, Matplotlib, and Python.

## Project origin

This is an independently organized portfolio implementation based on concepts studied in my undergraduate data science and machine learning coursework. It applies clustering, regression, and classification techniques to analyze student learning behavior using reusable Python workflows, independent analysis, and original documentation. It does not include assignment prompts, grading materials, or instructor-provided solution files.
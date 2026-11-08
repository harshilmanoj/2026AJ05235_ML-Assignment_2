# Phishing Website Classification

A machine learning classification project that compares five supervised learning algorithms for identifying legitimate and phishing websites. The project also includes a Streamlit application intended to let users upload test data, select a trained classifier, and review its evaluation results.

**Machine Learning Assignment 2**  

**Student Name:** Harshil Manoj  
**Student ID:** 2026AJ05235  

**Dataset:** UCI Phishing Websites  
**Task:** Binary classification  

---
## 1. Problem Statement

Phishing websites imitate legitimate websites to deceive users into sharing sensitive information such as login credentials, personal details, or financial information. Detecting these websites accurately is an important part of improving online security.

The objective of this project is to build and compare machine learning classification models that predict whether a website is **legitimate** or **phishing**, based on website and URL-related characteristics.

Five classification algorithms are implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC). The model comparison is used to identify the classifier that provides the strongest overall performance on the held-out test set.

## 2. Dataset Description

**Dataset name:** Phishing Websites  
**Source:** [UCI Machine Learning Repository – Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)

The dataset contains website-related attributes that can help distinguish legitimate websites from phishing websites.

| Dataset characteristic | Description |
|---|---:|
| Original number of instances | 11,055 |
| Number of input features | 30 |
| Number of target columns | 1 |
| Total columns | 31 |
| Classification type | Binary |
| Missing values | 0 |
| Duplicate records identified | 5,206 |
| Records retained after duplicate removal | 5,849 |

### Features

The 30 input features describe URL, domain, page, and website characteristics. Examples include:

- `having_ip_address`
- `url_length`
- `shortining_service`
- `having_at_symbol`
- `double_slash_redirecting`
- `prefix_suffix`
- `having_sub_domain`
- `sslfinal_state`
- `domain_registration_length`
- `favicon`
- `https_token`
- `request_url`
- `url_of_anchor`
- `links_in_tags`
- `web_traffic`
- `page_rank`
- `google_index`
- `links_pointing_to_page`
- `statistical_report`

The dataset contains 30 features in total; the examples above are not an exhaustive list.

### Target variable

The target column is named `target` in the project files:

| Target value | Class |
|---:|---|
| `-1` | Legitimate |
| `1` | Phishing |

In the original dataset, the target distribution is:

| Class | Number of instances |
|---|---:|
| Legitimate (`-1`) | 4,898 |
| Phishing (`1`) | 6,157 |

### Data preparation and split

The notebook identified **5,206 duplicate rows** and removed them before model training. The resulting dataset contained **5,849 records**.

The data was split into training and testing sets using an 80:20 stratified split with `random_state=42`.

| Split | Instances |
|---|---:|
| Training set | 4,679 |
| Test set | 1,170 |
| Total after duplicate removal | 5,849 |

Stratification was used to preserve the class proportions in both sets. The training and test sets each contain both legitimate and phishing examples.

Standardization is applied through pipelines for Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes are used without a scaling step in the notebook.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/harshilmanoj/2026AJ05235_ML-Assignment_2/

The repository contains the application source code, trained model files, test dataset, dependency list, and this README.

## 4. Models Used

The following five classification models were trained and evaluated using the same training and test split.

### Evaluation metrics

The metrics below are calculated for the **phishing class (`1`)** for Precision, Recall, and F1-score. AUC is calculated using the predicted probability of the phishing class. Accuracy and MCC are calculated using the complete binary classification results.

| Metric | Description |
|---|---|
| Accuracy | Proportion of all predictions that are correct. |
| AUC | Area under the ROC curve; measures how well the model separates the two classes across classification thresholds. |
| Precision | Of the websites predicted as phishing, the proportion that are actually phishing. |
| Recall | Of the actual phishing websites, the proportion correctly identified by the model. |
| F1-score | Harmonic mean of phishing-class Precision and Recall. |
| MCC | Matthews Correlation Coefficient; summarizes the quality of binary predictions using all four confusion-matrix outcomes. |

### Model comparison

The following values are taken from the notebook's held-out test-set evaluation. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9128 | 0.9738 | 0.9000 | 0.9223 | 0.9110 | 0.8258 |
| Decision Tree | 0.9171 | 0.9219 | 0.9180 | 0.9099 | 0.9139 | 0.8340 |
| K-Nearest Neighbors | 0.9205 | 0.9726 | 0.9246 | 0.9099 | 0.9172 | 0.8409 |
| Naive Bayes | 0.6547 | 0.9635 | 0.9939 | 0.2880 | 0.4466 | 0.4122 |
| Support Vector Machine | **0.9402** | **0.9827** | 0.9247 | **0.9541** | **0.9391** | **0.8808** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieved 91.28% accuracy and a 0.9110 phishing-class F1-score. Its phishing recall of 0.9223 indicates that it identifies a good proportion of phishing websites. Its results are balanced, although its overall metrics are lower than those of SVM and KNN in this experiment. |
| **Decision Tree** | Achieved 91.71% accuracy and a 0.9139 F1-score. Its phishing precision (0.9180) and recall (0.9099) are relatively balanced. Its AUC of 0.9219 is the lowest among the five models, indicating comparatively weaker ranking/separation performance in this evaluation. |
| **K-Nearest Neighbors (KNN)** | Achieved 92.05% accuracy, a 0.9172 F1-score, and a 0.8409 MCC. Its phishing precision is 0.9246, while its recall is 0.9099. It performs consistently across the reported metrics and is the second-best model by phishing-class F1-score. |
| **Naive Bayes** | Achieved 99.39% phishing precision, but its phishing recall is only 0.2880. It correctly identifies relatively few of the actual phishing websites: the confusion matrix shows 163 true positives and 403 false negatives. Consequently, its phishing-class F1-score (0.4466), accuracy (0.6547), and MCC (0.4122) are substantially lower than those of the other models. |
| **Support Vector Machine (SVM)** | Achieved the best overall results in this comparison: 94.02% accuracy, 0.9827 AUC, 0.9391 phishing-class F1-score, and 0.8808 MCC. Its phishing recall of 0.9541 is the highest among the five models, while its precision is 0.9247. It also has the lowest number of false negatives (26), making it the strongest model in this experiment for identifying phishing websites. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the overall winner, achieving the highest Accuracy (94.02%), AUC (0.9827), Recall (0.9541), F1-score (0.9391), and MCC (0.8808). It correctly identifies 540 of the 566 actual phishing websites and has the fewest false negatives (26), making it the strongest model among the five evaluated classifiers. |

### Overall winner for this dataset

**Support Vector Machine (SVM)** is the overall winner based on the held-out test-set results.

It achieved the highest Accuracy, AUC, Recall, F1-score, and MCC among the five models. Its phishing recall of 0.9541 means it correctly identified 540 of the 566 actual phishing websites in the test set. It produced 26 false negatives, fewer than any of the other evaluated models.

KNN was the second-best model by phishing-class F1-score, while Naive Bayes showed a strong precision but a substantial weakness in phishing recall.

## 5. Streamlit Application

The Streamlit application is designed to demonstrate the trained classifiers using an uploaded test CSV file.

**Live Streamlit App:** https://2026aj05235-ml-assignment2.streamlit.app/

### Application features

The application should provide the following assignment-required functionality:

- **Test dataset upload:** Upload a CSV file containing all required input features and the `target` column.
- **Model selection:** Select one of the five saved classification models.
- **Evaluation metrics:** Display Accuracy, AUC, Precision, Recall, F1-score, and MCC for the selected model.
- **Confusion matrix or classification report:** Display the classification results for the uploaded test data.
- **Model comparison:** Compare all five trained classification models side by side using evaluation metrics, performance charts, and a consolidated leaderboard to identify the best-performing model.
- **Data explorer:** Explore the uploaded dataset through interactive visualizations, feature distributions, class distribution, descriptive statistics, correlation heatmaps, and feature-level insights.


The application loads the saved model and evaluates it on the uploaded test data. It does **not** retrain the model on the uploaded test data.

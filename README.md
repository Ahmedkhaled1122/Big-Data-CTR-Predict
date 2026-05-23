# Taobao CTR Prediction — Big Data Pipeline

**Developed by:** Ahmed

## 📌 Project Overview
This project builds a complete end-to-end Big Data Pipeline to predict the Click-Through Rate (CTR) for advertisements on the **Taobao** platform. Due to the massive scale of the data (over 28 million records), the project utilizes distributed computing with **Apache Spark (PySpark)** and **Hadoop**, culminating in an interactive web application built with **Streamlit** to evaluate model performance.

## 📊 Dataset Details
The dataset consists of three main files:
* `raw_sample.csv`: 26,557,961 records of ad display logs.
* `ad_feature.csv`: 846,811 records of ad features.
* `user_profile.csv`: 1,061,768 records of user profiles.
* **Total Records:** 28,466,540
* **Target Variable (`clk`):** Binary classification (1 = clicked, 0 = not clicked). 
* **Note:** The dataset is highly imbalanced, with clicks representing only **5.14%** of the data.

## 🛠️ Tech Stack & Tools
* **Big Data Frameworks:** PySpark, Hadoop
* **Programming Languages & Libraries:** Python, Pandas, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn, Plotly
* **Web UI:** Streamlit

## 📁 Project Structure
1. **`EDA.ipynb.ipynb`**: Exploratory Data Analysis (EDA). Includes schema inspection, record counting, and target distribution analysis.
2. **`02_Preprocessing_1.ipynb`**: Data cleaning and preprocessing pipeline (Handling Missing Values → Removing Duplicates → Joining Tables → Feature Engineering → Encoding → Scaling → Imbalance).
3. **`03_Modeling.ipynb`**: Developing, training, and evaluating three machine learning models using Spark MLlib.
4. **`app.py`**: An interactive Streamlit web application that provides a comprehensive comparison of model performance and displays feature importance.

## 🚀 Model Performance
Three models were trained and evaluated. The results are as follows:

| Model | Train AUC-ROC | Test AUC-ROC | Test F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.8226 | 0.8263 | 0.8361 |
| **Random Forest** | 0.8309 | 0.8342 | 0.7629 |
| **Gradient Boosted Trees (GBT) ** | 0.8359 | **0.8397** | **0.7999** |

*The **Gradient Boosted Trees (GBT)** model achieved the highest Test AUC-ROC score, making it the best performing model for this pipeline.*

## ⚙️ How to Run the App

1. Ensure you have **Hadoop** and **Apache Spark** configured on your local machine.
2. Install the required Python dependencies:
   ```bash
   pip install pyspark pandas scikit-learn streamlit matplotlib seaborn plotly findspark

# 🚀 Machine Learning Monitoring Dashboard

This project builds a **Streamlit-based dashboard** covering the **full machine learning lifecycle**, including ETL, Model Training, Monitoring (with Evidently AI), and Explainability (with SHAP values).

The app contains **4 main interactive pages**:

---


---

## 📋 Dashboard Pages Overview

### 1. ETL Page - Local Data Visualization
- 🔹 Connects to a **local SQLite database**.
- 🔹 Displays the **loaded data** in a clean tabular format.
- 🔹 Includes basic **EDA (Exploratory Data Analysis)** and **visualizations**.
![image](https://github.com/user-attachments/assets/584891e9-7ecb-44e8-9662-eef4475ea988)
![image](https://github.com/user-attachments/assets/625d419d-02a6-4651-b321-bb8be5b9ddf3)

### 2. Model Training Page
- 🤖 Allows users to **train a machine learning model** based on the uploaded or extracted data.
- 📈 Shows **training metrics** such as:
  - Accuracy
  - Precision
  - Recall
  - Confusion matrix
- 💾 Saves the **trained model artifacts** for later inference and monitoring.

### 3. Monitoring Page - Evidently AI Reports
- 📊 Monitors **Data Drift**, **Concept Drift**, and **Model Drift** using **Evidently AI**.
- 📈 Generates interactive reports:
  - Data stability
  - Feature importance drift
  - Target drift and performance evaluation
- 🛡 Helps identify **model degradation over time**.

### 4. Explainable AI Page - SHAP Values
- 🧠 Uses **SHAP** to explain **model predictions**.
- 🎯 Visualizes feature contributions through:
  - SHAP Summary Plots
  - SHAP Force Plots
  - SHAP Dependence Plots
- 📜 Enhances model transparency and interpretability.

---

## 🛠 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/ritu-pixel/DockNest.git
cd DockNest
```
2. Install the required packages:

```bash

pip install -r requirements.txt
```
Run the Streamlit app:

```bash

streamlit run dashboard/app.py
```
(Optional): To run using Docker:

```bash


docker-compose up --build
```

##Technologies Used
Python 3.9+

Streamlit

SQLite

Scikit-learn

Evidently AI

SHAP

Docker

 Future Improvements
Integrate CI/CD pipelines for automatic deployments.

Add authentication for secured dashboards.

Set up scheduled monitoring for drift detection.

Deploy to AWS EC2, DockerHub, or Streamlit Community Cloud.

🙌 Author
Made with ❤️ by Ritu Rajput

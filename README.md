# 🛒 Customer Intelligence Platform

## End-to-End E-Commerce Analytics & Decision Engine

### Overview

Customer Intelligence Platform is a full-stack Data Science project that transforms raw e-commerce data into actionable business insights.

Built using transactional, behavioral, and review data, the platform enables customer segmentation, churn risk analysis, product health monitoring, recommendation generation, review intelligence, and funnel analytics through an interactive dashboard.

### Business Questions Solved

* Who are the most valuable customers?
* Which customers are at risk of churn?
* Which products are performing well or poorly?
* What are customers complaining about?
* Which products are frequently purchased together?
* Where do customers drop off in the purchase journey?

---

## Dataset

| Dataset     | Records |
| ----------- | ------- |
| Users       | 10,000  |
| Orders      | 20,000  |
| Order Items | 43,525  |
| Products    | 2,000   |
| Reviews     | 15,000  |
| Events      | 80,000  |

Total records processed: **170,000+**

---

## Features

### Executive Dashboard

* Revenue tracking
* Order analytics
* Customer metrics
* Category performance

### Customer Analytics

* RFM Segmentation
* Customer Lifetime Value (CLV)
* Churn Risk Prediction

### Product Analytics

* Product Health Score
* Top and Bottom Products
* Category Performance Monitoring

### Review Intelligence

* Sentiment Analysis
* Aspect-Based Sentiment Analysis
* Complaint Detection
* SHAP Explainability

### Recommendation Engine

* Frequently Bought Together
* User-Based Product Recommendations

### Behavioral Analytics

* Conversion Funnel Analysis
* Cart Abandonment Tracking
* Event Trend Analysis

---

## Machine Learning Models

### Sentiment Analysis

* TF-IDF Vectorization
* Logistic Regression
* Explainable predictions using SHAP

### Churn Prediction

* Random Forest Classifier
* Customer behavioral and transactional features

### Product Health Engine

Health Score combines:

* Sales Performance
* Review Ratings
* Sentiment Performance
* Complaint Rate

to generate a single business-friendly score between 0 and 100.

---

## System Architecture

Raw Data

Users
Orders
Order Items
Products
Reviews
Events

↓

Data Processing

Pandas + NumPy

↓

SQLite Data Warehouse

↓

Machine Learning Layer

* Sentiment Analysis
* Churn Prediction
* Product Health Scoring

↓

Interactive Streamlit Dashboard

---

## Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* SHAP

### Database

* SQLite

### Visualization

* Plotly

### Dashboard

* Streamlit

### Model Persistence

* Joblib

---

## Installation

```bash
git clone <repository-url>

cd customer-intelligence-platform

pip install "numpy<2"

pip install -r requirements.txt
```

Build warehouse:

```bash
python build_warehouse.py
```

Launch dashboard:

```bash
streamlit run app.py
```

---

## Key Outcomes

* Built an end-to-end analytics platform on 170K+ e-commerce records.
* Designed customer segmentation, churn prediction, recommendation, and sentiment analysis pipelines.
* Developed an interactive business intelligence dashboard using Streamlit.
* Implemented explainable AI using SHAP for transparent sentiment predictions.
* Created a unified SQLite warehouse integrating transactional, behavioral, and review data.

---

## Future Improvements

* DistilBERT-based sentiment analysis
* XGBoost churn prediction
* BERTopic review topic modeling
* FastAPI deployment
* Docker containerization
* PostgreSQL migration

---

## Author

Pavan

B.Tech Mechanical Engineering

IIITDM Kancheepuram

Aspiring Data Scientist | Machine Learning Enthusiast

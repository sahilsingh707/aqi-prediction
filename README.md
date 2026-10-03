# AQI Prediction 🌍

A Machine Learning project that predicts Air Quality Index (AQI) and classifies air quality based on environmental data. The project aims to make air quality information easier to understand and help users explore pollution patterns.

## 📌 Project Overview

Air pollution is a major environmental concern that affects human health and quality of life. This project uses machine learning models to analyze air quality data and predict pollution levels.

The project includes:

* **AQI Prediction:** Predicting AQI values using a regression model.
* **Air Quality Classification:** Classifying air quality using a classification model.
* **Data Analysis:** Exploring air quality data to understand pollution patterns.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Machine Learning
* Pickle for model serialization

## 📂 Project Structure

```text
aqi-prediction/
├── app.py
├── aqi_classifier.pkl
├── aqi_regressor.pkl
├── city_day.csv
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/sahilisingh707/aqi-prediction.git
cd aqi-prediction
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
streamlit run app.py
```

The application should open in your browser if Streamlit starts successfully.

## 🤖 Machine Learning Models

This project uses two trained machine learning models:

* **Regression Model:** Predicts numerical AQI values.
* **Classification Model:** Predicts air quality categories.

The models are saved as `.pkl` files and loaded by the application.

## 🎯 Project Objectives

* Explore air quality data and pollution trends.
* Apply machine learning to an environmental problem.
* Build an interactive application for AQI prediction.
* Demonstrate practical implementation of machine learning using Python.

## 🚀 Future Improvements

* Add visualizations of historical air pollution trends.
* Compare multiple machine learning algorithms.
* Improve model evaluation and prediction accuracy.
* Integrate updated air quality data.
* Add explainability features to understand which variables influence predictions.

## 👨‍💻 Author

**Sahil Singh**

B.Tech — Computer Science and Engineering (Artificial Intelligence)

GitHub: [sahilisingh707](https://github.com/sahilisingh707)

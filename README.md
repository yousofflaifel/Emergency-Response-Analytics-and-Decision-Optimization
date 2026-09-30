# Emergency Response Analytics and Decision Optimization

A data analytics and machine learning project that uses emergency incident data to analyze historical response performance, predict emergency response time, and optimize operational decision-making using a Genetic Algorithm.

> [!IMPORTANT]
> **The dataset used in this project is synthetic and is not real emergency-service data.**
>
> It was created/used for academic and demonstration purposes. It does not represent actual emergency incidents, ambulance operations, hospitals, patients, dispatch decisions, or real operational performance. The results should not be interpreted as real-world emergency response statistics.
> It can't be downloaded because it's very large. 

## Small Project Description

**An end-to-end data analytics solution for emergency response operations that combines descriptive analytics, predictive machine learning, and Genetic Algorithm optimization to understand response patterns, predict response time, and support dispatch decision-making.**

## Project Overview

The project follows a three-stage analytics approach:

1. **Descriptive Analytics** — understand what happened.
2. **Predictive Analytics** — estimate what may happen.
3. **Prescriptive Analytics** — determine an optimized course of action.

The overall workflow is:

```text
Synthetic Emergency Response Dataset
                 |
                 v
        Data Cleaning & Preparation
                 |
                 v
        Descriptive Analytics
                 |
                 v
          Feature Selection
                 |
                 v
        Predictive Analytics
                 |
        +--------+--------+--------+
        |        |        |        |
        v        v        v        v
     Linear    Decision   KNN      ANN
   Regression   Tree    Regression
        |        |        |        |
        +--------+--------+--------+
                 |
                 v
       Model Evaluation & Comparison
                 |
                 v
       Genetic Algorithm Optimization
                 |
                 v
     Dispatch Decision Support
```

## Data Disclaimer

This project uses a **synthetic dataset** for academic purposes.

The dataset should not be treated as real emergency-service data. The incident records, response times, operational conditions, resource availability, and other variables are simulated data used to demonstrate data analytics and machine learning techniques.

The project uses emergency response operations as the business context for the analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- PyGAD
- Jupyter Notebook
- Machine Learning
- Artificial Neural Networks
- Genetic Algorithms
- Statistical Analysis
- Data Visualization

The project uses Python-based analytics tools including Pandas, NumPy, Seaborn, Scikit-learn, TensorFlow/Keras, and PyGAD. 

## Dataset

The final prepared dataset contains:

```text
Records: 368,065
Features: 25
Target: Response_Time
```

The dataset contains operational, environmental, incident, and emergency-response variables.

### Main Variables

| Variable | Description |
|---|---|
| Timestamp | Time of the emergency incident |
| Incident Severity | Severity classification of the incident |
| Incident Type | Accident, Cardiac Arrest, Fire, or Other |
| Region Type | Urban, Suburban, or Rural |
| Traffic Congestion | Low, Moderate, or High |
| Weather Condition | Clear, Rainy, or Stormy |
| Ambulance Availability | Availability of ambulances |
| Hospital Capacity | Available beds in nearby hospitals |
| Distance to Incident | Distance to the incident in kilometers |
| Number of Injuries | Number of injured individuals |
| Specialist Availability | Availability of medical specialists |
| Road Type | Highway, Local Road, or Unpaved Road |
| Emergency Level | Minor, Major, or Critical |
| Ambulance Speed | Ambulance speed in km/h |
| Fuel Level | Remaining ambulance fuel percentage |
| Weather Impact | No Impact, Moderate, or Severe |
| Dispatch Coordinator | Human or AI |
| Response Time | Estimated emergency response time in minutes |

The metadata defines response time as the estimated time for emergency units to respond and describes the operational ranges for variables such as distance, injuries, ambulance speed, hospital capacity, and fuel level.

## Data Preparation

The data preparation process includes:

- Dataset inspection
- Data quality assessment
- Removing the timestamp from the modeling dataset
- Encoding categorical variables
- Preparing numerical features
- Selecting `Response_Time` as the prediction target
- Splitting the dataset into training and testing sets

The final modeling dataset uses an:

```text
80% Training
20% Testing
```

split.

One-hot encoding is applied to categorical variables so that they can be used by the machine learning models.

## Descriptive Analytics

Descriptive analytics is used to understand historical emergency response patterns.

The analysis includes:

- Summary statistics
- Frequency analysis
- Histograms
- Count plots
- Correlation analysis
- Incident-type analysis
- Emergency-level analysis
- Weather analysis
- Operational-factor analysis

### Response Time

The dataset analysis reports:

```text
Mean:   15.06 minutes
Median: 15.01 minutes
Minimum: 5 minutes
Maximum: 30 minutes
```

The response-time distribution is approximately centered around 15 minutes.

Because the dataset is synthetic, these values are only characteristics of the academic dataset and do not represent actual emergency-service performance.

## Incident Analysis

The project analyzes the distribution of emergency incidents.

### Incident Types

The dataset contains:

- Accident
- Cardiac Arrest
- Fire
- Other

Accidents represent approximately half of the incidents in the dataset, while cardiac arrest and other incidents each represent approximately one-fifth, and fire incidents approximately one-tenth.

### Emergency Levels

The emergency levels are:

```text
Minor
Major
Critical
```

The reported distribution is approximately:

```text
Minor:    69.95%
Major:    19.99%
Critical: 10.06%
```

Again, these percentages describe the synthetic academic dataset only.

## Feature Selection

Feature selection was performed to identify variables that are relevant to predicting emergency response time and to reduce unnecessary model complexity.

Two feature-selection techniques were compared:

### SelectKBest

SelectKBest uses statistical scores to rank features.

The analysis identified variables including:

- Weather Condition
- Specialist Availability
- Road Type
- Dispatch Coordinator

### Random Forest Feature Importance

Random Forest feature importance was used to identify variables based on their contribution to the ensemble model.

Important variables identified included:

- Ambulance Speed
- Fuel Level
- Distance
- Hospital Capacity
- Number of Injuries

### Feature Selection Decision

The project compared the performance of the selected feature sets using Linear Regression.

The reported results showed similar overall performance between the two approaches. SelectKBest was selected for the remaining predictive models because it provided comparable performance while being simpler to interpret.

## Predictive Analytics

The predictive stage aims to estimate:

```text
Response_Time
```

using operational and environmental variables.

The following regression techniques were evaluated:

- Linear Regression
- Decision Tree Regression
- KNN Regression
- Artificial Neural Network

## Linear Regression

Linear Regression was used as a baseline regression model and also played a role in validating the selected feature set.

The model was evaluated using:

- MAE
- MSE
- RMSE
- R²

The analysis reported that Linear Regression produced the lowest prediction error among the tested predictive approaches.

## Decision Tree Regression

A Decision Tree Regression model was used to capture potentially non-linear relationships between emergency-response variables and response time.

The model was compared with Linear Regression using the same evaluation metrics.

## KNN Regression

K-Nearest Neighbors Regression was evaluated as another prediction technique.

The results indicated that KNN produced the weakest predictive performance among the tested models.

This suggests that neighboring observations did not contain enough useful information to reliably predict response time for this dataset.

## Artificial Neural Network

An Artificial Neural Network was also evaluated to determine whether a more complex non-linear model could improve response-time prediction.

The ANN produced results similar to the Decision Tree model and did not significantly improve prediction performance over the simpler approaches.

## Model Evaluation

The project evaluates regression models using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual response times.

### Mean Squared Error (MSE)

Penalizes larger prediction errors more heavily.

### Root Mean Squared Error (RMSE)

Provides an error measurement in the same unit as response time.

### R²

Measures how much of the variation in the target variable is explained by the model.

## Predictive Model Findings

The project found that:

- Linear Regression produced the lowest prediction error among the tested models.
- Decision Tree and ANN produced similar results.
- KNN produced the weakest performance.
- Increasing model complexity did not significantly improve prediction accuracy.

The analysis also identified a key limitation: the predictor variables showed weak relationships with `Response_Time`, which limited the ability of more complex models to discover strong predictive patterns.

This demonstrates an important data-science principle:

> Better features and better data can be more important than increasing model complexity.

## Prescriptive Analytics

The final stage of the project uses a **Genetic Algorithm (GA)** to move from prediction toward decision support.

The goal is to determine weighting coefficients that minimize emergency response time while respecting operational constraints.

The Genetic Algorithm was implemented using:

```text
PyGAD
```

## Genetic Algorithm Optimization

The optimization model uses decision variables related to:

```text
Distance
Number of Injuries
Hospital Capacity
Fuel Level
```

The Genetic Algorithm searches for suitable coefficients for these variables.

The objective is to minimize the estimated response time while maintaining the specified constraints.

The approach demonstrates how optimization can be used after predictive analysis to support operational decisions.

## Analytics Framework

The project follows:

```text
Descriptive Analytics
"What happened?"
        |
        v
Predictive Analytics
"What is likely to happen?"
        |
        v
Prescriptive Analytics
"What should we do?"
```

### Descriptive

Identifies historical patterns, trends, relationships, and operational bottlenecks.

### Predictive

Uses machine learning to estimate future response-time behavior.

### Prescriptive

Uses Genetic Algorithm optimization to identify suitable operational coefficients for dispatch decision support.

## Key Questions

The project was designed around four main questions:

1. What operational patterns exist in emergency response data?
2. Which factors influence response performance?
3. Can future response times be predicted?
4. What operational strategy can be optimized using the available variables?

## How to Run

### Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow pygad jupyter
```

### Run the Project

1. Clone or download the repository.
2. Place the dataset in the expected project directory.
3. Open the Jupyter Notebook or Python script containing the implementation.
4. Start Jupyter:

```bash
jupyter notebook
```

5. Run the data preparation cells first.
6. Run the descriptive analytics section.
7. Run feature selection.
8. Train and evaluate the predictive models.
9. Run the Genetic Algorithm optimization.

## Project Results

The project demonstrates a complete analytics pipeline:

```text
Raw Emergency Data
        |
        v
Data Preparation
        |
        v
Descriptive Analytics
        |
        v
Feature Selection
        |
        v
Predictive Modelling
        |
        v
Model Comparison
        |
        v
Genetic Algorithm
        |
        v
Decision Support
```

The project shows that predictive performance depends strongly on the relevance and quality of input variables. More complex models did not automatically produce better predictions when the available predictors had weak relationships with response time.

## Limitations

The most important limitation is that the dataset is **synthetic**.

Therefore:

- The response-time statistics do not represent actual emergency-service performance.
- The model metrics should not be interpreted as real-world predictive performance.
- The discovered relationships may not exist in real emergency-response systems.
- The Genetic Algorithm recommendations are an academic optimization exercise.
- The system has not been validated using real operational emergency-service data.
- Real deployment would require domain validation, additional operational variables, and appropriate safety and governance procedures.

## Future Improvements

Potential future improvements include:

- Using real, properly governed emergency-response data.
- Adding traffic and road-network data.
- Adding GPS-based ambulance locations.
- Including real-time weather information.
- Including ambulance workload and availability history.
- Adding hospital distance and capacity in real time.
- Improving feature engineering.
- Testing ensemble regression models.
- Hyperparameter optimization.
- Explainable AI for predictive models.
- Real-time dispatch optimization.
- Integrating the optimization model into an operational decision-support system.

## Learning Outcomes

This project demonstrates practical experience with:

- Data analytics
- Descriptive analytics
- Predictive analytics
- Prescriptive analytics
- Data preprocessing
- Exploratory data analysis
- Feature selection
- Linear Regression
- Decision Tree Regression
- KNN Regression
- Artificial Neural Networks
- Regression evaluation
- MAE
- MSE
- RMSE
- R²
- Genetic Algorithms
- PyGAD
- Data visualization
- Decision support
- Optimization

## Academic Context

This project was developed as a university **Data Analytics Assignment** under the title:

```text
Emergency Response Analytics and Decision Optimization
```

The objective was to demonstrate how historical operational data can be transformed into descriptive insights, predictive models, and optimization-based decision support.

## Author

**Yousof Flaifel**

Data Science & AI Student  
Al Hussein Technical University (HTU)

---

University Data Analytics Project

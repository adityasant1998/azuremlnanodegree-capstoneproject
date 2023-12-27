# Heart Failure Predictions Using Microsoft Azure ML Studio

This project is part of the Udacity Azure ML Nanodegree. The Heart Failure Clinical Records Dataset is used to train a machine learning model that predicts mortality by heart failure. The model is developed using Hyper Drive and Auto ML methods, and the model with the highest accuracy is retrieved (voting ensemble in this case) and deployed in the cloud with Azure Container Instances (ACI) as a webservice, with authentication enabled. Once the model is deployed, the endpoint’s behavior is analyzed by getting a response from the service and logs are retrieved at the end. The dataset is publicly available on Kaggle. Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Most cardiovascular diseases can be prevented by addressing behavioral risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity, and harmful use of alcohol using population-wide strategies. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management wherein a machine learning model can be of great help. 

## Dataset

### Overview
This Dataset is available publicy in https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

### Task
The task of this project is to train a model to predict whether the person with cardiovascular disease will survive or not. To access the dataset, two different methods were used for two different models.

### Access

For HyperDrive Approach: Dataset from a public repository, loaded with TabularDataset.
For AutoML Approach: Dataset registered from local files, loaded from Azure workspace.

## Automated ML Approach and Results


## Hyperparameter Tuning Approach and Results



## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording



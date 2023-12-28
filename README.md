# Heart Failure Predictions Using Microsoft Azure ML Studio

This project is part of the Udacity Azure ML Nanodegree. The Heart Failure Clinical Records Dataset is used to train a machine learning model that predicts mortality by heart failure. The model is developed using Hyper Drive and Auto ML methods, and the model with the highest accuracy is retrieved (voting ensemble in this case) and deployed in the cloud with Azure Container Instances (ACI) as a webservice, with authentication enabled. Once the model is deployed, the endpoint’s behavior is analyzed by getting a response from the service and logs are retrieved at the end. 
## Dataset


### Overview
This Dataset is available publicy in https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
The dataset is publicly available on Kaggle. Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Most cardiovascular diseases can be prevented by addressing behavioral risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity, and harmful use of alcohol using population-wide strategies. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management wherein a machine learning model can be of great help. 

### Aim
The task of this project is to train a model to predict whether the person with cardiovascular disease will survive or not. To access the dataset, two different methods were used for two different models.

### Impact and Significance
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help

### Access
For HyperDrive Approach: Dataset from a public repository, loaded with TabularDataset.
For AutoML Approach: Dataset registered from local files, loaded from Azure workspace.

## Data 
The data contains 299 records of the below mentioned attributes:

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/b4be3055ebbf256e251f9d28c974c8f69ede1587/additional_screenshots/data%20exploration.jpg)






## Automated ML Approach and Results
Tired of manually tweaking machine learning models? Say hello to AutoML: your shortcut to automated model development! AutoML takes the pain out of repetitive tasks, letting Azure Machine Learning train and optimize models based on your desired metric.

Here's how it works for our heart failure prediction project:

Data Prep:

We register the Heart_Failure_Clinical_Records_Dataset.csv from our local files into the Azure workspace.
Before converting it to a pandas dataframe, we create the workspace, experiment, and compute cluster.
AutoML Configuration:

To run AutoML smoothly, we define an AutoMLConfig with specific settings and compute target.
This includes setting the experiment timeout to 30 minutes, task type to Classification (predicting categories of new data), primary metric to accuracy (model effectiveness), and other parameters like label column, cross-validations, training dataset, allowed concurrent iterations, and automatic feature engineering.
Model Arsenal:

AutoML taps into a variety of models like LightGBM, SVM, XGBoost, Random Forest, and ensemble approaches like Voting and Stacking.
It also pre-processes data using various techniques like scaling and normalization to enhance model performance.
Visualization and Model Selection:

Once the experiment runs, we use the Notebook widget to track progress and visualize results.
Upon successful completion, the best model is retrieved and saved for further use.
In summary, AutoML simplifies model building by automating tedious tasks and offering a powerful toolbox of algorithms and pre-processing techniques. This lets you focus on what matters most: interpreting results and applying the best model to solve your problem.

Here are the screenshots of steps that I took to use AutoML in Azure ML studio:

![automl configuration](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/1.automl%20configuration.jpg)

![automl rundetails](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/2.automl%20rundetails.jpg)

![automl rundetails 2](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/3.automl%20rundetails%202.jpg)

![automl best model accuracy](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/4.automl%20best%20model%20accuracy.jpg)

![automl best run](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/5.automl%20best%20run.jpg)

![automl completion](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/6.automl%20completion.jpg)

### Results

![data guardrails](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/7.data%20guardrails.jpg)

![Auto ML Rundetails model comparison](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/8.Auto%20ML%20Rundetails%20model%20comparison.jpg)

![model performance](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/9.model%20performance.jpg)

![dataset explorer](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/10.dataset%20explorer.jpg)

![aggregate feature importance](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/11.aggregate%20feature%20importance.jpg)

![metrics](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/12.metrics.jpg)

![metrics](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/13.metrics.jpg)

![metrics](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/14.metrics.jpg)



## Hyperparameter Tuning Approach and Results

Data Acquisition and Preparation:

The heart failure dataset is fetched from a public GitHub repository using TabularDatasetFactory.
It's split into training (70%) and testing (30%) sets for model development and evaluation.
Model Selection and Training:

Logistic regression, a popular classification algorithm, is chosen for predicting heart failure outcomes.
The training script train.py is used to train the model with hyperparameter tuning via HyperDrive.
HyperDrive Configuration:

The workspace, experiment, and cluster are set up in Azure Machine Learning.
Bandit policy is applied to optimize resource usage by terminating underperforming runs.
Random sampling explores different combinations of hyperparameters:
Inverse of regularization parameter (C): (0.1, 1)
Maximum number of iterations (max_iter): (50, 100, 150, 200)
The goal is to maximize accuracy, with a maximum of 10 total runs and 4 concurrent runs.
Experiment Execution and Visualization:

The experiment is submitted and visualized using the RunDetails widget.
Upon completion, the best model is saved in the output folder and registered for future use.

Here are the screenshots of steps that I took to use Hyperparameter Tuning Approach in Azure ML studio:

![hyperdrive configuration](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/15.hyperdrive%20configuration.jpg)

![hyperdrive run](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/16.hyperdrive%20run.jpg)

![hyperdrive run_id](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/17.hyperdrive%20run_id.jpg)

### Result

![hyperdrive best model accuracy](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/18.hyperdrive%20best%20model%20accuracy.jpg)


## Model Deployment

AutoML outperformed HyperDrive in terms of accuracy, achieving a best run accuracy of 87.63% compared to HyperDrive's 75.75%.
AutoML's best model was deployed to ensure optimal performance in real-world applications.
Deployment process:

### Environment Details

I have created a custom mlvenv.yml file as shown below, I have pasted this is the default directory of my workspace (./mlvenv.yml)

Model Registration and Download:

The best AutoML model was selected and registered for deployment.
Necessary files associated with the model were downloaded for the deployment process.
Environment and Inference Creation:

An environment was set up with the required conda dependencies to support model execution.
A score.py script was included, defining initialization and exit functions for the model.
This environment served as the foundation for inference (prediction generation).
Deployment with Azure Container Instance:

The model and its environment were deployed using Azure Container Instance (ACI), a flexible and scalable containerization service.
Resources were allocated with 1 CPU core and 1 GB of memory to accommodate model execution.
Monitoring and Verification:

Application Insights was enabled to monitor the deployed model's performance and health.
The service's state was verified to ensure successful deployment and readiness for use.
Conclusion:

By following these steps, the AutoML model with the highest accuracy was successfully deployed, ready to make predictions and deliver value in the real world.

## Model Deployment 

Demo of model deployment : https://drive.google.com/file/d/1k1wi2BsmIyk0KqbzSzf4l64VJnJPeP6P/view?usp=sharing

![model deployment](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/19.model%20deployment.jpg)

![pipeline testing](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/2c479a64c21efd64225fc71580620e3ea8a23363/screenshots/20.pipeline%20testing.jpg)

## Screen Recording

The screen recording can be accessed from the following link : https://drive.google.com/file/d/1yBbH6BEeuYApoB0_A54PGJcCNxOEsdYL/view?usp=sharing

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

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/data%20attribute%20list.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/data%20exploration%20ui.jpg)


Understanding Features:

1. age: displays the age of the individual.

2. sex: displays the gender of the individual using the following format :

• 1 = male

• 0 = female

3. cp (Chest-Pain Type): displays the type of chest-pain experienced by the individual using the following format :

• 0 = typical angina

• 1 = atypical angina

• 2= non — anginal pain

• 3 = asymptotic

4. trestbps(Resting Blood Pressure): displays the resting blood pressure value of an individual in mmHg (unit)

5. chol(Serum Cholestrol): displays the serum cholesterol in mg/dl (unit)

6. fbs (Fasting Blood Sugar): compares an individual's fasting blood sugar value with 120mg/dl.

• If fasting blood sugar > 120mg/dl then : 1 (true) else : 0 (false)

7. restecg (Resting ECG): displays resting electrocardiographic results • 0 = normal

• 1 = having ST-T wave abnormality

• 2 = left ventricular hyperthrophy

8. thalach(Max Heart Rate Achieved): displays the max heart rate achieved by an individual.

9. exang (Exercise induced angina):

• 1 = yes

• 0 = no

10.oldpeak (ST depression induced by exercise relative to rest): displays the value of an integer or float.

11.slope (Peak exercise ST segment) :

• 0 = upsloping

• 1 = flat

• 2 = downsloping

12.ca (Number of major vessels (0–3) colored by fluoroscopy): displays the value as integer or float.

13.thal: displays the thalassemia (is an inherited blood disorder that causes your body to have less hemoglobin than normal) :

• 0 = normal

• 1 = fixed defect

• 2 = reversible defect

14.target (Diagnosis of heart disease): Displays whether the individual is suffering from heart disease or not :

• 0 = absence

• 1 = present

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

### Data Ingestion Method

Following are the steps that I followed to get data :

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata1.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata2.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata3.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata4.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata5.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata6.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata7.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/getdata8.jpg)

Now I have successfully ingested the data, and it can be seen in the data tab in the UI.

Here are the screenshots of steps that I took to use AutoML in Azure ML studio:

![automl configuration](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/1.automl%20configuration.jpg)

#### Config Explanation:
experiment_timeout_minutes: 30

This sets a time limit of 30 minutes for the experiment to run. After this duration, the experiment will stop, even if it hasn't fully completed.

2. task: 'classification'

This specifies the type of machine learning task. In this case, it's a classification task, meaning the model will aim to predict a category or class for each data point.

3. primary_metric: 'accuracy'

This determines the main metric that will be used to evaluate and compare different models during the experiment. Here, accuracy will be used, which measures the proportion of correct predictions made by the model.

4. training_data: dataset

This references the dataset that will be used to train the machine learning models. The 'dataset' variable likely holds the actual dataset you've prepared for training.

5. label_column_name: 'DEATH_EVENT'

This identifies the column in the dataset that contains the target values or labels that the model will try to predict. The model will learn to associate patterns in other columns with the values in this 'DEATH_EVENT' column.

6. n_cross_validations: 5

This sets the number of cross-validation folds to use during training. Cross-validation helps assess model performance and prevent overfitting. Using 5 folds means the dataset will be split into 5 parts, and the model will be trained and evaluated 5 times, each time using a different part for evaluation.

7. max_concurrent_iterations: 4

This controls the maximum number of iterations (model training attempts) that can run concurrently. Allowing for 4 concurrent iterations can potentially speed up the experiment.

8. featurization: 'auto'

This enables automatic featurization, which means Azure AutoML will handle feature engineering for you. It will analyze the data and create appropriate features for the models to use.

###  Rundetails and Output of AutoML Model : 

![automl rundetails](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/49ee367ae2f9eebc466de82eb8dfcec0dd7cecb7/screenshots/2.automl%20rundetails.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/2762e97c4347650b681a402625ba2977a7e52494/additional_screenshots/output%20of%20automl.jpg)

### Top Models Overview :

Take a look are details of top 2 models that I got from AutoML:

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/automl%20best%20models.jpg)

#### VotingEnsemble :

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/automl%20mode1%20ensemble%20details.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/accuracy1.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/accuracy%20table%201.jpg)

#### StackEnsemble

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/automl%20model2%20ensemble%20details.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/accuracy2.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/f08569237a3fba3b0b28f4025eec27ddd40c332f/additional_screenshots/accuracy%20table%202.jpg)


Here are several factors that could explain why VotingEnsemble performed better than StackEnsemble in your Azure AutoML run:

1. Nature of the Data and Problem:

Relatively simple relationships: VotingEnsemble often excels when the relationships between features and the target variable are reasonably straightforward. It can effectively combine diverse model predictions in such cases.
Limited dataset size: Stacking might benefit more from larger datasets to achieve optimal meta-model training. If your dataset is relatively small, VotingEnsemble might be less prone to overfitting.

2. Model Diversity and Strength:

Diverse individual models: VotingEnsemble benefits from having a set of individual models that make different types of errors. This diversity can lead to better combined performance.
Similarly strong individual models: If the individual models in the ensemble have relatively similar performance levels, VotingEnsemble might be sufficient for effective aggregation.

3. Meta-Model Choice in Stacking:

Suboptimal meta-model: The performance of StackEnsemble heavily relies on the meta-model's ability to learn how to best combine the base model predictions. If the chosen meta-model isn't well-suited to the task, it could limit performance.

4. Hyperparameter Tuning:

Tuning for StackEnsemble: Stacking often involves more hyperparameters to tune, including those for the base models and the meta-model. If hyperparameter tuning wasn't as thorough for StackEnsemble, it might not have reached its full potential.

Additional Considerations:

Experiment timeout: If the experiment timeout limited the exploration of different stacking configurations, VotingEnsemble might have had an advantage.

Cross-validation folds: The specific cross-validation folds used could have impacted model performance, potentially favoring VotingEnsemble in this particular run.


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

Here's a breakdown of the HyperDriveConfig settings in Azure AutoML SDK:

1. estimator:

This specifies the base estimator or machine learning algorithm that will be used for hyperparameter tuning. It's the model whose hyperparameters will be explored to find the best configuration.

2. hyperparameter_sampling:

This defines the method for sampling different hyperparameter combinations during the tuning process. Common options include random sampling, grid search, or Bayesian optimization.

3. policy:

This sets the early termination policy, which allows for stopping poorly performing runs early to save time and resources. The early_termination_policy variable likely holds a specific early stopping configuration.

4. primary_metric_name:

This identifies the key metric that will be used to evaluate and compare different hyperparameter combinations. In this case, it's set to "Accuracy."

5. primary_metric_goal:

This specifies whether the goal is to maximize or minimize the primary metric. Here, PrimaryMetricGoal.MAXIMIZE means the tuning process will aim to find hyperparameters that produce the highest possible accuracy.

6. max_total_runs:

This sets the maximum number of hyperparameter tuning runs that will be executed. The tuning process will stop after completing 10 runs.

7. max_concurrent_runs:

This controls the maximum number of runs that can execute concurrently. Allowing for 4 concurrent runs can potentially speed up the tuning process by leveraging multiple compute resources.

#### Explanation of parameters used in train.py training script:

1. Argument Parsing:

The code sets up argument parsing using argparse to allow for customization of hyperparameters when running the script.
It defines two arguments:
--C: Controls the inverse of regularization strength in the logistic regression model. Smaller values lead to stronger regularization, which can help prevent overfitting.
--max_iter: Sets the maximum number of iterations the model will run during training. This limits training time and can prevent excessive optimization attempts.
2. Logging Hyperparameters:

The run.log statements are likely used to record the chosen hyperparameter values for tracking and analysis purposes, potentially within a logging framework or experiment tracking tool.
3. Model Creation and Training:

The code instantiates a LogisticRegression model with the specified C and max_iter values.
It then trains the model using the fit method on the provided training data (x_train and y_train).
4. Evaluation:

The code calculates the model's accuracy on a test set (x_test and y_test) using the score method.
The accuracy score is logged for reference and evaluation purposes.
Key Points:

The code trains a logistic regression model, a common algorithm for classification tasks.
It allows for customization of regularization strength (C) and maximum iterations (max_iter) through command-line arguments.
It logs hyperparameter values and accuracy for tracking and analysis.

![hyperdrive run](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/16.hyperdrive%20run.jpg)

![hyperdrive run_id](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/17.hyperdrive%20run_id.jpg)

### Result

![hyperdrive best model accuracy](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/18.hyperdrive%20best%20model%20accuracy.jpg)


## Model Deployment

AutoML outperformed HyperDrive in terms of accuracy, achieving a best run accuracy of 89% compared to HyperDrive's 85%.
AutoML's best model was deployed to ensure optimal performance in real-world applications.
Deployment process:

### Environment Details

I have created a custom mlvenv.yml file as shown below, I have pasted this is the default directory of my workspace (./mlvenv.yml)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/0a3068173a41f9e16c3b2e5883bac93ccd523cd8/additional_screenshots/mlvenv.jpg)

Used in Hyperdrive approach (code in notebook):

estimator = SKLearn(source_directory = '.', compute_target=cpu_cluster_name, entry_script='train.py', conda_dependencies_file = './mlvenv.yml')

AutoML Approach:

Used : Python 3.8 AzureML for notebook

But for deployment used in following code in notebook :

env = Environment.from_conda_specification('mlvenv', './mlvenv.yml')

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

## Model Deployment through SDK:

Demo of model deployment : https://drive.google.com/file/d/1k1wi2BsmIyk0KqbzSzf4l64VJnJPeP6P/view?usp=sharing

![model deployment](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/358aa9ba2cc90365db400763f8f6dec7b08114cf/screenshots/19.model%20deployment.jpg)

These steps are done in below screenshot : 
Prepare a sample input data (JSON or another format as required by your model).
Use the requests library to send an HTTP POST request to your model's endpoint.
Display the response from the model to confirm it's receiving and processing input as expected.
![pipeline testing](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/2c479a64c21efd64225fc71580620e3ea8a23363/screenshots/20.pipeline%20testing.jpg)

## Model Deployment through UI:

Select best model and select deploy:

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/0a3068173a41f9e16c3b2e5883bac93ccd523cd8/additional_screenshots/deployment%20using%20webui1.jpg)

![](https://github.com/adityasant1998/azuremlnanodegree-capstoneproject/blob/0a3068173a41f9e16c3b2e5883bac93ccd523cd8/additional_screenshots/deployment%20using%20webui2.jpg)

## Screen Recording of full project

The screen recording can be accessed from the following link : https://drive.google.com/file/d/1yBbH6BEeuYApoB0_A54PGJcCNxOEsdYL/view?usp=sharing

## How to improve the project in the future

1. Ensemble Exploration:

Experiment with different ensemble types: I should try various ensemble methods beyond VotingEnsemble and StackEnsemble, such as BlendingEnsemble or BaggingEnsemble. Different methods might prove more effective depending on my dataset and problem.
Optimize individual models: I should focus on improving the performance of the base models within the ensemble. Their strength significantly impacts the overall ensemble performance.
Explore weighted voting: I should assign different weights to individual models in VotingEnsemble based on their perceived reliability or performance.

2. Hyperparameter Tuning:

Devote more time to tuning: I should allocate sufficient time and resources for thorough hyperparameter tuning, especially for StackEnsemble, as it often involves more hyperparameters to explore.
Consider advanced tuning techniques: I should employ Bayesian optimization or other advanced tuning methods that can efficiently explore the hyperparameter space.
Tune both model types: I should tune both VotingEnsemble and StackEnsemble to discover their optimal configurations and fairly compare their potential.

3. Data Augmentation:

Increase dataset size (if possible): If feasible, I should collect more data or employ data augmentation techniques to enlarge the dataset. This can often improve model performance, especially for more complex models like StackEnsemble.

4. Feature Engineering:

Investigate feature importance: I should analyze feature importance to identify the most influential features and potentially remove irrelevant or redundant ones.
Create new features: I should explore feature engineering techniques to generate new, informative features that might enhance model performance.

5. Meta-Model Exploration (for StackEnsemble):

Try different meta-models: I should experiment with various meta-models for StackEnsemble to see if any lead to better performance. Linear models, tree-based models, or even neural networks could be viable options for the meta-model.

6. Evaluation and Monitoring:

Use a held-out test set: I should always evaluate model performance on a separate test set that hasn't been used during training or validation to get a reliable estimate of generalization.
Track performance over time: I should monitor model performance in production to detect potential degradation and trigger retraining if necessary.

7. Consider Interpretability:

If interpretability is crucial: If understanding model behavior is important for my application, VotingEnsemble might be a more interpretable choice compared to StackEnsemble.

# Student Performance Indicator

## Problem Statement

```WE WILL ANALYSE THE PERFORMANCE OF STUDENTS BASED VARIOUS FEATURES SUCH AS GENDER , PARENT'S BACKGROUND AND STATUS OF TEST PREPARATIONS.```

## Data Dictionary

* gender : sex of students -> (Male/female)
* race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
* parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
* lunch : having lunch before test (standard or free/reduced)
* test preparation course : complete or not complete before test
* math score
* reading score
* writing score


### Create project template hierarchy
```bash
python template.py
```

### Setup development environment
```bash
bash init_setup.sh
```

### Acivate environment
```bash
source activate ./env
```

### Install project as local package
```bash
pip install -r requirement.txt
```

## Pipelines
### Training Pipeline
    * Data Ingestion (fetched data from source)
    * Data Transformation (Feature Engineering, Data Preprocessing)
    * Model Builing (Create a model using the processed data)

## MLFlow & DagsHub
Copy the values from DagsHub > Repo > Remote > Experiments

```bash
set MLFLOW_TRACKING_URI=<>
set MLFLOW_TRACKING_USERNAME=<>
set MLFLOW_TRACKING_PASSWORD<>
```
If the above are not set, then ML Experiments gets registered in local system else gets published to DagsHub

#### Command to train the pipeline
```bash
python src\pipeline\training_pipeline.py
```

### Prediction Pipeline
    * Two types of prediction pipeline
        * Single record prediction
        * Batch prediction


## Streamlit App
```bash
streamlit run app.py
```

## Deployment of DockerImage on AWS
* AWS - ECR
* AWS - AppRunner

## Cloud Deployed Links
* https://gemstonepriceprediction.streamlit.app/
* https://g3smncimby.us-east-1.awsapprunner.com/


## Dataset Link
* https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction
* https://raw.githubusercontent.com/abhijitpaul0212/DataSets/main/gemstone.csv

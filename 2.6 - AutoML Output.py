# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **labcluster-616237** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/576707025419958/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/576707025419957) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _10.4.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "rating"
time_col = "year"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC > **NOTE:** The dataset loaded below is a sample of the original dataset.
# MAGIC Pyspark's [sample](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sample.html) method
# MAGIC is used to sample this dataset.
# MAGIC <br/>
# MAGIC > Rows were sampled with a sampling fraction of **0.8461812622184512**

# COMMAND ----------

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("9853e33d8aab4817972bdf55f898b002", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["Short", "votes", "year", "Comedy", "Drama", "Action", "Animation", "length", "Documentary", "Romance", "budget", "title", "mpaa"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), ["budget", "length", "votes", "year"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers, sparse_threshold=0))
])

transformers.append(("numerical", numerical_pipeline, ["length", "budget", "year", "votes"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, sparse_threshold=0, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers.append(("onehot", one_hot_pipeline, ["Action", "Animation", "Comedy", "Documentary", "Drama", "Romance", "Short", "mpaa"]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Medium-cardinality categoricals
# MAGIC Convert each medium-cardinality categorical column into a numerical representation.
# MAGIC Each string column is hashed to 1024 float columns.
# MAGIC Each numeric column is imputed with zeros.

# COMMAND ----------

from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

imputers = {
}

for col in ["title"]:
    hasher = FeatureHasher(n_features=1024, input_type="string")
    if col in imputers:
        imputer_name, imputer = imputers[col]
    else:
        imputer_name, imputer = "impute_string_", SimpleImputer(fill_value='', missing_values=None, strategy='constant')
    hash_pipeline = Pipeline(steps=[
        (imputer_name, imputer),
        (f"{col}_hasher", hasher),
    ])
    transformers.append((f"{col}_pipeline", hash_pipeline, [col]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Given that year is provided as the `time_col`, the data will be split based on time order.
# The most recent data will be used as validation set, and thus shuffling when splitting the data is inappropriate.
df_loaded = df_loaded.sort_values(time_col).reset_index(drop=True)

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, shuffle=False)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, shuffle=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/576707025419958/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from lightgbm import LGBMRegressor

help(LGBMRegressor)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

lgbmr_regressor = LGBMRegressor(
  colsample_bytree=0.4027484472323993,
  lambda_l1=134.95276610609923,
  lambda_l2=470.62819691802224,
  learning_rate=0.034704605022147145,
  max_bin=202,
  max_depth=4,
  min_child_samples=55,
  n_estimators=2113,
  num_leaves=10,
  subsample=0.7771181269559879,
  random_state=715957211,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("regressor", lgbmr_regressor),
])

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
])

mlflow.sklearn.autolog(disable=True)
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(experiment_id="576707025419958", run_name="lightgbm") as mlflow_run:
    model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__eval_set=[(X_val_processed,y_val)], regressor__verbose=False)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    lgbmr_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    lgbmr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    lgbmr_val_metrics = {k.replace("val_", ""): v for k, v in lgbmr_val_metrics.items()}
    lgbmr_test_metrics = {k.replace("test_", ""): v for k, v in lgbmr_test_metrics.items()}
    display(pd.DataFrame([lgbmr_val_metrics, lgbmr_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None

import lightgbm
from mlflow.tracking import MlflowClient

lgbmr_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(lgbmr_temp_dir)
lgbmr_client = MlflowClient()
lgbmr_model_env_path = lgbmr_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", lgbmr_temp_dir)
lgbmr_model_env_str = open(lgbmr_model_env_path)
lgbmr_parsed_model_env_str = yaml.load(lgbmr_model_env_str, Loader=yaml.FullLoader)

lgbmr_parsed_model_env_str["dependencies"][-1]["pip"].append(f"lightgbm=={lightgbm.__version__}")

with open(lgbmr_model_env_path, "w") as f:
  f.write(yaml.dump(lgbmr_parsed_model_env_str))
lgbmr_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=lgbmr_model_env_path, artifact_path="model")
shutil.rmtree(lgbmr_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

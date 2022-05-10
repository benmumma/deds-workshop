# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC This notebook performs exploratory data analysis on the dataset.
# MAGIC To expand on the analysis, attach this notebook to the **labcluster-616237** cluster,
# MAGIC edit [the options of pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html), and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/576707025419958/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/576707025419957) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC 
# MAGIC Runtime Version: _10.4.x-cpu-ml-scala2.12_

# COMMAND ----------

# MAGIC %md
# MAGIC > **NOTE:** The dataset loaded below is a sample of the original dataset.
# MAGIC Pyspark's [sample](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sample.html) method
# MAGIC is used to sample this dataset.
# MAGIC <br/>
# MAGIC > Rows were sampled with a sampling fraction of **0.8461812622184512**

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import databricks.automl_runtime

from mlflow.tracking import MlflowClient

# Download input data from mlflow into a pandas DataFrame
# Create temporary directory to download data
temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(temp_dir)

# Download the artifact and read it
client = MlflowClient()
training_data_path = client.download_artifacts("9853e33d8aab4817972bdf55f898b002", "data", temp_dir)
df = pd.read_parquet(os.path.join(training_data_path, "training_data"))

# Delete the temporary data
shutil.rmtree(temp_dir)

target_col = "rating"

# Convert columns detected to be of semantic type categorical
categorical_columns = ["Action", "Animation", "Comedy", "Drama", "Documentary", "Romance", "Short"]
df[categorical_columns] = df[categorical_columns].applymap(str)

# Convert columns detected to be of semantic type numeric
numeric_columns = ["budget"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Type Detection Alerts
# MAGIC 
# MAGIC For details about the definition of the semantic types and how to override the detection, see
# MAGIC [Databricks documentation on semantic type detection](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl#semantic-type-detection).
# MAGIC 
# MAGIC - Semantic type `categorical` detected for columns `Action`, `Animation`, `Comedy`, `Documentary`, `Drama`, `Romance`, `Short`. Training notebooks will encode features based on categorical transformations.
# MAGIC - Semantic type `numeric` detected for column `budget`. Training notebooks will convert each column to a numeric type and encode features based on numerical transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

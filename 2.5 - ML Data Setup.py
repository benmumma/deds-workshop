# Databricks notebook source
#Check the data sets available:

display(dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/ggplot2"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Dataframe

# COMMAND ----------


datapath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/movies.csv"
moviesDF = spark.read.format("csv")\
              .option("header","true")\
              .option("inferschema","true")\
              .load(datapath)


# COMMAND ----------

userhome = "dbfs:/user/ben.mumma@databricks.com"
delta_path = userhome + "/delta2/movies/"

# COMMAND ----------

moviesDF.write.mode("overwrite").format("delta").save(delta_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Database and Delta Table

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Create a Database and set to use */
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS dbacademy_benmumma;
# MAGIC USE dbacademy_benmumma

# COMMAND ----------

#Create the table to use for our AutoML Experiment
spark.sql("""
  CREATE TABLE IF NOT EXISTS movies_delta 
  USING DELTA 
  LOCATION '{}' 
""".format(delta_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check out the Data

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from movies_delta

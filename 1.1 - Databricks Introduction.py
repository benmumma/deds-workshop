# Databricks notebook source
# MAGIC %md
# MAGIC ## ![test](https://redislabs.com/wp-content/uploads/2016/12/lgo-partners-databricks-125x125.png) Databricks Introduction Workshop
# MAGIC 
# MAGIC Databricks is a **Unified Analytics Platform**, from the original creators of Apache Spark™, that unifies data science and engineering across the Machine Learning lifecycle from data preparation, to experimentation and deployment of ML applications. It's a place where data engineers, data scientists, and business analysts can run data ETL workloads, conduct data exploration, and make business decisions in their own environment!

# COMMAND ----------

# DBTITLE 1,Headers & Markdowns
# MAGIC %md
# MAGIC this is text
# MAGIC 
# MAGIC this is `code`
# MAGIC 
# MAGIC ### This is a Header
# MAGIC #### Header 4
# MAGIC ##### Header 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Magic Commands
# MAGIC Databricks cotains several different magic commands.
# MAGIC 
# MAGIC #### Mix Languages
# MAGIC You can override the default language by specifying the language magic command `%<language>` at the beginning of a cell. Switch between Python, R, SQL, and Scala within the same notebook The supported magic commands are:  
# MAGIC * `%python`
# MAGIC * `%r`
# MAGIC * `%scala`
# MAGIC * `%sql`
# MAGIC 
# MAGIC #### Auxiliary Magic Commands
# MAGIC * `%sh`: Allows you to run shell code in your notebook. To fail the cell if the shell command has a non-zero exit status, add the -e option. This command runs only on the Apache Spark driver, and not the workers. 
# MAGIC * `%fs`: Allows you to use dbutils filesystem commands.
# MAGIC * `%md`: Allows you to include various types of documentation, including text, images, and mathematical formulas and equations.
# MAGIC 
# MAGIC #### Other Magic Commands
# MAGIC * `%pip`: Allows you to easily customize and manage your Python packages on your cluster
# MAGIC 
# MAGIC #### Multi-Cursor
# MAGIC Multi-Cursor can be activated with Command + mouse left click (ctrl + alt + left click on window) on all the lines that you want to select to edit multiple lines at once

# COMMAND ----------

print("Hello Python!")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'Hello SQL'

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC print("Hello R!", quote=FALSE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Revision History Tracking 
# MAGIC From within a notebook, click the right-most Revision History button to see all past versions of the notebook. 
# MAGIC 
# MAGIC Click Restore if you want to revert the notebook to a previous version

# COMMAND ----------

# MAGIC %md
# MAGIC # Collaboration
# MAGIC Try this:
# MAGIC 1. Highlight code
# MAGIC 2. A small bubble should appear on the right-hand side
# MAGIC 3. Any code (Python, markdown, etc.) can be commented this way

# COMMAND ----------

# MAGIC %md
# MAGIC ##Databricks File System - DBFS
# MAGIC * DBFS is a layer over a cloud-based object store
# MAGIC * Files in DBFS are persisted to the object store
# MAGIC * The lifetime of files in the DBFS are **NOT** tied to the lifetime of our cluster
# MAGIC * Mounting other object stores into DBFS gives Databricks users access via the file system (one of many techniques for pulling data into Spark)
# MAGIC 
# MAGIC ### Databricks Utilities - dbutils
# MAGIC * You can access the DBFS through the Databricks Utilities class (and other file IO routines).
# MAGIC * An instance of DBUtils is already declared for us as `dbutils`.
# MAGIC * For in-notebook documentation on DBUtils you can execute the command `dbutils.help()`.
# MAGIC 
# MAGIC ### Magic Command: &percnt;fs
# MAGIC **&percnt;fs** is a wrapper around `dbutils.fs`, thus `dbutils.fs.ls("/databricks-datasets")` is equivalent to running `%fs ls /databricks-datasets`

# COMMAND ----------

# DBTITLE 1,Use %fs to programmatically list files in the file system and visualize the incoming data
# MAGIC %fs ls /databricks-datasets/

# COMMAND ----------

# DBTITLE 1,Select and display the entire incoming csv data using a simple SQL SELECT query
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM csv.`/databricks-datasets/asa/airlines/2008.csv`

# COMMAND ----------

dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/ggplot2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Function
# MAGIC Databricks supports various types of visualizations out of the box using the `display` function. The `display(..)` command is overloaded with a lot of other capabilities:
# MAGIC * Presents up to 1000 records.
# MAGIC * Exporting data as CSV.
# MAGIC * Rendering a multitude of different graphs.
# MAGIC * Rendering geo-located data on a world map.
# MAGIC 
# MAGIC And as we will see later, it is also an excellent tool for previewing our data in a notebook.
# MAGIC 
# MAGIC #### DataFrames
# MAGIC The easiest way to create a Spark DataFrame visualization in Databricks is to call `display(<dataframe-name>)`.  `Display` also supports Pandas DataFrames.
# MAGIC 
# MAGIC 💡If you see `OK` with no rendering after calling the `display` function, mostly likely the DataFrame or collection you passed in is empty.
# MAGIC 
# MAGIC #### Images
# MAGIC display renders columns containing image data types as rich HTML. display attempts to render image thumbnails for DataFrame columns matching the Spark ImageSchema. Thumbnail rendering works for any images successfully read in through the spark.read.format('image') function. More info [here](https://docs.databricks.com/notebooks/visualizations/index.html#images).
# MAGIC 
# MAGIC #### Visualizations
# MAGIC The display function supports a rich set of plot types that can be configured by clicking the bar chart icon ![bar](https://docs.databricks.com/_images/chart-button.png):
# MAGIC 
# MAGIC ![charts](https://docs.databricks.com/_images/display-charts.png)

# COMMAND ----------

display(dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/ggplot2"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## A note on Temporary Views
# MAGIC Temporary views are session-scoped and are dropped when session ends because it skips persisting the definition in the underlying metastore.  These are a great way to simplify SQL queries, switch easily between languages to perform quick analysis, develop a visualization, etc.  Note: These do not help performance as they are lazily executed
# MAGIC 
# MAGIC Creating a temporary view:
# MAGIC * python: `df.createOrReplaceTempView("<NAME>")`
# MAGIC * R (SparkR): `createOrReplaceTempView(df, "<NAME>")`
# MAGIC * SQL: `CREATE [ OR REPLACE ] [ [ GLOBAL ] TEMPORARY ] VIEW [ IF NOT EXISTS ] view_identifier
# MAGIC     create_view_clauses AS query`
# MAGIC 
# MAGIC ⚠️ **Note**: These do not help performance as they are lazily executed

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Exploration

# COMMAND ----------

#dbutils.help()  list available utilities along with a short description for each utility
dbutils.fs.help() # list available commands for a utility along with a short description of each command
dbutils.fs.help("cp") # display help for a command, run .help("<command-name>") after the command name.

# COMMAND ----------

username = 'ben_mumma'
spark.sql(f"CREATE DATABASE IF NOT EXISTS dbacademy_{username}")
spark.sql(f"USE dbacademy_{username}")

# COMMAND ----------

# Creating a dataframe
# maintain headers and schema
# Transformation
datapath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/movies.csv"
moviesDF = spark.read.format("csv")\
              .option("header","true")\
              .option("inferschema","true")\
              .load(datapath)

# COMMAND ----------

moviesDF.count()

# COMMAND ----------

# Action (execution happens here)
display(moviesDF)
# moviesDF.show()
#download option (not recommended more than 1 mn rows)

# COMMAND ----------

# Different functionality within the cell
# Plotting and options
display(moviesDF)

# COMMAND ----------

# MAGIC %md ## Use widgets in a notebook
# MAGIC 
# MAGIC Databrick utilites (e.g. `dbutils`) provides functionality for many common tasks within Databricks notebooks: 
# MAGIC https://docs.databricks.com/dev-tools/databricks-utils.html
# MAGIC 
# MAGIC One useful feature is "Widgets" that allow you to dynamically program within your notebooks: https://docs.databricks.com/notebooks/widgets.html

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.dropdown("mpaa", "PG", ["PG", "PG-13", "R", "NC-17"])

# COMMAND ----------

display(moviesDF.filter(moviesDF.mpaa == getArgument('mpaa')))

# COMMAND ----------

# MAGIC %md
# MAGIC # Delta Tables
# MAGIC Delta Lake is an open source storage layer that brings reliability to data lakes.
# MAGIC 
# MAGIC Delta Benefits:
# MAGIC 
# MAGIC * **ACID transactions on Spark:** Serializable isolation levels ensure that readers never see inconsistent data.
# MAGIC * **Scalable metadata handling:** Leverages Spark distributed processing power to handle all the metadata for petabyte-scale tables with billions of files at ease.
# MAGIC * **Streaming and batch unification:** A table in Delta Lake is a batch table as well as a streaming source and sink. Streaming data ingest, batch historic backfill, interactive queries all just work out of the box.
# MAGIC * **Schema enforcement:** Automatically handles schema variations to prevent insertion of bad records during ingestion.
# MAGIC * **Time travel:** Data versioning enables rollbacks, full historical audit trails, and reproducible machine learning experiments.
# MAGIC * **Upserts and deletes:** Supports merge, update and delete operations to enable complex use cases like change-data-capture, slowly-changing-dimension (SCD) operations, streaming upserts, and so on.

# COMMAND ----------

userhome = "dbfs:/user/ben.mumma@databricks.com"
delta_path = userhome + "/delta2/movies/"

# COMMAND ----------

moviesDF.write.mode("overwrite").format("delta").save(delta_path)

# COMMAND ----------

spark.sql("""
  CREATE TABLE IF NOT EXISTS movies_delta 
  USING DELTA 
  LOCATION '{}' 
""".format(delta_path))

# COMMAND ----------

# DBTITLE 1,magic commands
# MAGIC %sql
# MAGIC SELECT * FROM movies_delta where rating > 9

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY movies_delta

# COMMAND ----------

# MAGIC %sql 
# MAGIC DELETE from movies_delta where votes < 100

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(_c0) from movies_delta

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY movies_delta

# COMMAND ----------

# MAGIC %sql 
# MAGIC RESTORE TABLE movies_delta TO VERSION AS OF 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(_c0) from movies_delta

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://databricks.com/wp-content/uploads/2019/08/Delta-Lake-Multi-Hop-Architecture-Bronze.png)

# COMMAND ----------

dbutils.widgets.get("X")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Format SQL Code
# MAGIC Databricks provides tools that allow you to format SQL code in notebook cells quickly and easily. These tools reduce the effort to keep your code formatted and help to enforce the same coding standards across your notebooks.
# MAGIC 
# MAGIC You can trigger the formatter in the following ways:
# MAGIC * Keyboard shortcut: Press **Cmd+Shift+F**.
# MAGIC * Command context menu: Select **Format SQL** in the command context drop-down menu of a SQL cell. This item is visible only in SQL notebook cells and those with a `%sql` language magic.
# MAGIC ![Format SQL](https://docs.databricks.com/_images/notebook-formatsql-cmd-context.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Quickly Share your Findings
# MAGIC After your exploratory data science work, share your findings via: 
# MAGIC 1. Dashboard: In the top utility bar, click the drop-down to create a new dashboard. Or click the small chart icon at the top-right of each cell
# MAGIC 2. Export a notebook to HTML to be viewed in any browser. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Do's and Don'ts
# MAGIC Below are just a couple of tips for how to improve performance
# MAGIC * Use Spark SQL functions
# MAGIC * Don’t use Python UDFs (use Pandas UDFs)
# MAGIC * Avoid .toPandas() on large datasets
# MAGIC * Avoid .collect() - used to retrieve the data from the Dataframe
# MAGIC * Avoid for loops or row-by-row operations
# MAGIC * Checkpoint large joins, or cache
# MAGIC * Split some jobs which have very different characteristics
# MAGIC * Size your clusters - check the Spark UI
# MAGIC * Avoid disk spill
# MAGIC * Benchmark your application (CPU vs memory)
# MAGIC * Use latest Databricks Runtime Versions
# MAGIC * Run optimize (or enable auto-optimize) on Delta Tables

# COMMAND ----------

# DBTITLE 1,Additional Learning Resources
# MAGIC %md
# MAGIC <table>
# MAGIC <tr>
# MAGIC   
# MAGIC   <td>
# MAGIC     <a href="https://community.databricks.com?utm_source=databricks&utm_medium=web&utm_campaign=7014N0000026zJyQAI" target="_blank">
# MAGIC      Ask the Community<br/>
# MAGIC       <img src="https://databricks.com/wp-content/uploads/2021/09/db-comm-blog-og.png" width=300/>
# MAGIC     </a>
# MAGIC   </td> 
# MAGIC   <td>
# MAGIC      <a href="https://databricks.com/learn?utm_source=databricks&utm_medium=web&utm_campaign=7014N0000026zJyQAI" target="_blank">
# MAGIC        Learn about training and more<br/>
# MAGIC       <img src="https://databricks.com/wp-content/uploads/2021/08/learn-thumbnail-1.jpg" width=300/>
# MAGIC      </a>
# MAGIC   </td>
# MAGIC   <td>
# MAGIC     <a href="https://www.youtube.com/playlist?list=PLTPXxbhUt-YVPwG3OWNQ-1bJI_s_YRvqP" target="_blank">
# MAGIC       View tech talks from the wider community<br/>
# MAGIC       <img src="https://opengraph.githubassets.com/1ad323cfce5b3be45f235ac2eda8ab6900a9c00d42541f5dd7af5e762c84bfdc/databricks/tech-talks" width=300/>
# MAGIC     </a>
# MAGIC   </td>
# MAGIC 
# MAGIC   <td>
# MAGIC     <a href="https://www.youtube.com/watch?v=6Q8qPZ7c1O0&list=PLTPXxbhUt-YVAGgN0aNCqY4Jydg324X8u?utm_source=databricks&utm_medium=web&utm_campaign=7014N0000026zJyQAI" target="_blank">
# MAGIC       Watch product demos<br/>
# MAGIC       <img src="https://databricks.com/de/wp-content/uploads/2021/08/2021-02-Demo-Hub-OG-1200x628-1-min.jpeg" width=300/>
# MAGIC     </a>
# MAGIC   </td>
# MAGIC </tr>
# MAGIC </table>

# COMMAND ----------



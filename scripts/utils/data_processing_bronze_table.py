import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType


# the raw csv has TWO columns both called "Account", so we skip the header names and name the columns ourselves
RAW_COLUMNS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_received",
               "receiving_currency", "amount_paid", "payment_currency", "payment_format", "is_laundering"]


def process_bronze_table(snapshot_date_str, bronze_directory, spark, csv_file_path="data/raw/HI-Small_Trans.csv"):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # everything stays as text in bronze, types are fixed in silver
    schema = StructType([StructField(c, StringType(), True) for c in RAW_COLUMNS])

    # load data - IRL ingest from back end source system (daily file from core banking)
    # timestamp looks like "2022/09/01 00:20", keep only the transactions of this day
    df = spark.read.csv(csv_file_path, header=True, schema=schema)
    df = df.filter(F.substring(col("timestamp"), 1, 10) == snapshot_date.strftime("%Y/%m/%d"))
    row_count = df.count()
    print('transactions', snapshot_date_str, 'row count:', row_count)

    # skip if no data on this day
    if row_count == 0:
        print('no data, skipped')
        return df

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_transactions_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = bronze_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df

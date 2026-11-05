from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType

# the raw csv has TWO columns called "Account", so we skip the header names and set our own
RAW_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


def process_bronze_table(
    snapshot_date_str, bronze_directory, spark, csv_file_path="data/raw/HI-Small_Trans.csv"
):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # everything stays as text in bronze, types are fixed in silver
    schema = StructType([StructField(c, StringType(), True) for c in RAW_COLUMNS])

    # IRL this is the daily file from core banking, here we cut one day out of the full csv
    # timestamp looks like "2022/09/01 00:20"
    df = spark.read.csv(csv_file_path, header=True, schema=schema)
    df = df.filter(F.substring(col("timestamp"), 1, 10) == snapshot_date.strftime("%Y/%m/%d"))
    row_count = df.count()
    print("transactions", snapshot_date_str, "row count:", row_count)

    if row_count == 0:
        print("no data, skipped")
        return df

    partition_name = "bronze_transactions_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = bronze_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df

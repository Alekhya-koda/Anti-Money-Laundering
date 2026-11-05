import argparse
import os

import pyspark

import utils.data_processing_bronze_table

# run from the project folder:
#   python scripts/bronze_transactions.py --snapshotdate "2022-09-01"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder.appName("bronze_transactions")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    bronze_directory = "datamart/bronze/transactions/"
    os.makedirs(bronze_directory, exist_ok=True)

    utils.data_processing_bronze_table.process_bronze_table(snapshotdate, bronze_directory, spark)

    spark.stop()

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

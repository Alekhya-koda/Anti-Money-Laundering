import argparse
import os

import pyspark

import utils.data_processing_silver_table

# run from the project folder:
#   python scripts/silver_transactions.py --snapshotdate "2022-09-01"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder.appName("silver_transactions")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    bronze_directory = "datamart/bronze/transactions/"
    silver_directory = "datamart/silver/transactions/"
    dq_directory = "datamart/silver/data_quality/"
    for directory in [silver_directory, dq_directory]:
        os.makedirs(directory, exist_ok=True)

    utils.data_processing_silver_table.process_silver_table(
        snapshotdate, bronze_directory, silver_directory, dq_directory, spark
    )

    spark.stop()

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

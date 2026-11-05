import argparse
import os

import pyspark

import utils.data_processing_gold_table

# label = the account sends laundering in the 2 days FROM the snapshot date,
# so the labels of a snapshot are only complete 2 days later
# run from the project folder:
#   python scripts/gold_label_store.py --snapshotdate "2022-09-05"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder.appName("gold_label_store")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    silver_directory = "datamart/silver/transactions/"
    gold_label_store_directory = "datamart/gold/label_store/"
    os.makedirs(gold_label_store_directory, exist_ok=True)

    utils.data_processing_gold_table.process_labels_gold_table(
        snapshotdate, silver_directory, gold_label_store_directory, spark, label_days=2
    )

    spark.stop()

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

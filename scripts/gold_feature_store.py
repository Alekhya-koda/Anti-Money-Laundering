import argparse
import os
import pyspark

import utils.data_processing_gold_table

# features of a snapshot use the 4 days BEFORE the snapshot date
# to call this script (from the project folder): python scripts/gold_feature_store.py --snapshotdate "2022-09-05"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_feature_store") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create gold datalake
    silver_directory = "datamart/silver/transactions/"
    gold_feature_store_directory = "datamart/gold/feature_store/"

    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    # run data processing
    utils.data_processing_gold_table.process_features_gold_table(snapshotdate, silver_directory, gold_feature_store_directory, spark, lookback_days=4)

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)

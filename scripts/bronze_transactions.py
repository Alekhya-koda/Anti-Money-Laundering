import argparse
import os
import pyspark

import utils.data_processing_bronze_table

# to call this script (from the project folder): python scripts/bronze_transactions.py --snapshotdate "2022-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_transactions") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create bronze datalake
    bronze_directory = "datamart/bronze/transactions/"

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    # run data processing
    utils.data_processing_bronze_table.process_bronze_table(snapshotdate, bronze_directory, spark)

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

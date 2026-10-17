import argparse
import os

import utils.graph_features

# graph features (pagerank, payment loops, 2-hop fan in / fan out) on the same 4 day window as the feature store
# the graph fits in memory (~400k accounts, ~500k edges), so this job uses pandas + networkx instead of spark
# to call this script (from the project folder): python scripts/gold_graph_features.py --snapshotdate "2022-09-05"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # create gold datalake
    silver_directory = "datamart/silver/transactions/"
    gold_graph_feature_directory = "datamart/gold/graph_feature_store/"

    if not os.path.exists(gold_graph_feature_directory):
        os.makedirs(gold_graph_feature_directory)

    # run data processing
    utils.graph_features.process_graph_features_table(snapshotdate, silver_directory, gold_graph_feature_directory, lookback_days=4)

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)

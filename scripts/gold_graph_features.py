import argparse
import os

import utils.graph_features

# graph features on the same 4-day window as the feature store
# the graph fits in memory (~400k accounts, ~500k edges), so pandas + networkx instead of Spark
# run from the project folder:
#   python scripts/gold_graph_features.py --snapshotdate "2022-09-05"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    silver_directory = "datamart/silver/transactions/"
    gold_graph_feature_directory = "datamart/gold/graph_feature_store/"
    os.makedirs(gold_graph_feature_directory, exist_ok=True)

    utils.graph_features.process_graph_features_table(
        snapshotdate, silver_directory, gold_graph_feature_directory, lookback_days=4
    )

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

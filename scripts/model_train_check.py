import argparse
import os
import sys
import pprint
from datetime import datetime

from mlflow.tracking import MlflowClient

from utils.model_utils import MODEL_NAME, train_valid_snapshots, load_training_data

# runs every day before the 3 training tasks and decides if we train today:
# no champion yet, the champion is a week old, or the monitor raised a drift flag
#   exit 0  -> train today
#   exit 99 -> no training today, Airflow marks this task and the training tasks as skipped
# to call this script (from the project folder): python scripts/model_train_check.py --snapshotdate "2022-09-09"

SKIP_EXIT_CODE = 99


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["train_snapshot_date_str"], config["valid_snapshot_date_str"] = train_valid_snapshots(snapshotdate)
    config["model_name"] = MODEL_NAME
    config["retrain_every_days"] = 7
    config["retrain_flag_filepath"] = "datamart/gold/model_monitoring/retrain_flag.txt"
    pprint.pprint(config)

    # --- check if we need new models ---
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(config["model_name"], "champion")
    except Exception:
        champion = None

    if champion is None:
        print('no champion model yet, train the first models')
    elif os.path.exists(config["retrain_flag_filepath"]):
        print('monitor found drift, retrain')
    else:
        champion_train_date = datetime.strptime(champion.tags["model_train_date"], "%Y-%m-%d")
        days_old = (config["model_train_date"] - champion_train_date).days
        if days_old < config["retrain_every_days"]:
            print('champion version', champion.version, 'is', days_old, 'days old, no retrain today, skipped')
            return False
        print('champion is', days_old, 'days old, weekly retrain')

    # --- check that the train and valid snapshots have features and labels ---
    for snapshot_date_str in [config["train_snapshot_date_str"], config["valid_snapshot_date_str"]]:
        if load_training_data(snapshot_date_str) is None:
            print('snapshot', snapshot_date_str, 'not ready yet, no training today, skipped')
            return False

    print('train today')
    print('\n\n---completed job---\n\n')
    return True


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    if not main(args.snapshotdate):
        sys.exit(SKIP_EXIT_CODE)

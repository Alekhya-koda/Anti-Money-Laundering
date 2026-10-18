import argparse
import pprint

from utils.model_utils import train_valid_snapshots, load_training_data

# first task of aml_training_pipeline (run by hand): checks that the train and valid snapshots
# of this train date have features and labels, otherwise the training run stops here with an error
# to call this script (from the project folder): python scripts/model_train_check.py --snapshotdate "2022-09-09"


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["train_snapshot_date_str"], config["valid_snapshot_date_str"] = train_valid_snapshots(snapshotdate)
    pprint.pprint(config)

    # --- check that the train and valid snapshots have features and labels ---
    for snapshot_date_str in [config["train_snapshot_date_str"], config["valid_snapshot_date_str"]]:
        data_pdf = load_training_data(snapshot_date_str)
        if data_pdf is None:
            raise ValueError("snapshot " + snapshot_date_str + " has no features or labels yet, "
                             "run aml_data_pipeline further before training with train date " + snapshotdate)
        print('snapshot', snapshot_date_str, 'ok:', len(data_pdf), 'accounts,', int(data_pdf["label"].sum()), 'positives')

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)

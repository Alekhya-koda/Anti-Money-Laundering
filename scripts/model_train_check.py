import argparse
import pprint

from utils.model_utils import load_training_data, train_valid_snapshots

# first task of aml_training_pipeline: stops the run if the train / valid snapshots
# of this train date do not have features and labels yet
# run from the project folder:
#   python scripts/model_train_check.py --snapshotdate "2022-09-09"


def main(snapshotdate):
    print("\n\n---starting job---\n\n")

    config = {}
    config["model_train_date_str"] = snapshotdate
    config["train_snapshot_date_str"], config["valid_snapshot_date_str"] = train_valid_snapshots(
        snapshotdate
    )
    pprint.pprint(config)

    for snapshot_date_str in [
        config["train_snapshot_date_str"],
        config["valid_snapshot_date_str"],
    ]:
        data_pdf = load_training_data(snapshot_date_str)
        if data_pdf is None:
            raise ValueError(
                "snapshot " + snapshot_date_str + " has no features or labels yet, "
                "run aml_data_pipeline further before training with train date " + snapshotdate
            )
        n_positive = int(data_pdf["label"].sum())
        print(f"snapshot {snapshot_date_str} ok: {len(data_pdf)} accounts, {n_positive} positives")

    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

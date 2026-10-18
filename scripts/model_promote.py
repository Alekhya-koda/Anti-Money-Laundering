import argparse
import os
import pprint
import urllib.request
import pandas as pd
from datetime import datetime, timezone

from mlflow.tracking import MlflowClient

from utils.model_utils import MODEL_NAME, MODEL_ALIASES

# promote a shadow model to champion (or roll back to an older champion) in one step, run by a person
# after checking the shadow results. It moves the aliases, records who / when / why (tags on the model
# version + a row in promotion_log.csv) and tells the API to reload.
# to call this script (from the project folder):
#   python scripts/model_promote.py --version 6 --by "Zhou Jie" --reason "v6 beat v2 on 3 labelled days"


def promotion_plan(aliases, version, past_champions):
    # aliases: {alias: version} now in the registry, version: the version that becomes champion,
    # past_champions: versions that were champion before (from the promotion log)
    # returns (kind, aliases to set {alias: version}, aliases to delete)
    if version == aliases.get("champion"):
        raise ValueError("version " + version + " is already the champion")
    shadows = [alias for alias in MODEL_ALIASES[1:] if alias in aliases]

    if version in [aliases[alias] for alias in shadows]:
        # promotion: the other shadow model becomes challenger, the aliases of the retrain go away,
        # so it is one champion and one challenger again
        others = [aliases[alias] for alias in shadows if aliases[alias] != version]
        to_set = {"champion": version}
        if others:
            to_set["challenger"] = others[0]
        to_delete = [alias for alias in shadows if alias.startswith("shadow_")]
        if not others and "challenger" in aliases:
            to_delete.append("challenger")
        return "promotion", to_set, to_delete

    if version in past_champions:
        # rollback: the champion goes back to an older champion, the shadow models stay as they are
        return "rollback", {"champion": version}, []

    raise ValueError("version " + version + " is not a shadow model and was never champion. Shadow models: " +
                     (", ".join(alias + " v" + aliases[alias] for alias in shadows) or "none"))


def main(version, promoted_by, reason):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["model_name"] = MODEL_NAME
    config["version"] = version
    config["promoted_by"] = promoted_by
    config["reason"] = reason
    config["promotion_log"] = "datamart/gold/model_training/promotion_log.csv"
    config["api_url"] = os.getenv("API_URL", "http://api:8000")
    pprint.pprint(config)

    # --- check the request against the registry and the promotion log ---
    client = MlflowClient()
    aliases = client.get_registered_model(config["model_name"]).aliases
    print('aliases before:', aliases)
    log_pdf = None
    past_champions = set()
    if os.path.exists(config["promotion_log"]):
        log_pdf = pd.read_csv(config["promotion_log"], dtype=str)
        past_champions = set(log_pdf["previous_champion"])
    kind, to_set, to_delete = promotion_plan(aliases, version, past_champions)
    print(kind, 'of version', version)

    # --- move the aliases ---
    previous_champion = aliases.get("champion", "-")
    for alias, alias_version in to_set.items():
        client.set_registered_model_alias(config["model_name"], alias, alias_version)
        print(alias, '-> version', alias_version)
    for alias in to_delete:
        client.delete_registered_model_alias(config["model_name"], alias)
        print(alias, 'alias removed (was version', aliases[alias] + ')')
    aliases = client.get_registered_model(config["model_name"]).aliases
    print('aliases after:', aliases)

    # --- record who, when and why: tags on the new champion + one row in the promotion log ---
    promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tags = {"promoted_by": promoted_by, "promoted_at": promoted_at, "promoted_reason": reason,
            "previous_champion": previous_champion}
    for key, value in tags.items():
        client.set_model_version_tag(config["model_name"], version, key, value)

    row_pdf = pd.DataFrame([{
        "promoted_at": promoted_at,
        "kind": kind,
        "champion": version,
        "previous_champion": previous_champion,
        "challenger": aliases.get("challenger", "-"),
        "promoted_by": promoted_by,
        "reason": reason,
    }])
    if log_pdf is not None:
        row_pdf = pd.concat([log_pdf, row_pdf], ignore_index=True)
    os.makedirs(os.path.dirname(config["promotion_log"]), exist_ok=True)
    row_pdf.to_csv(config["promotion_log"], index=False)
    print('saved to:', config["promotion_log"])

    # --- tell the API, it also checks the champion alias every minute by itself ---
    try:
        request = urllib.request.Request(config["api_url"] + "/reload", method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            print('API:', response.read().decode())
    except Exception as e:
        print('could not call', config["api_url"] + "/reload", '(' + str(e) + '),',
              'the API picks the new champion up within a minute by itself')

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--version", type=str, required=True, help="model version that becomes champion")
    parser.add_argument("--by", type=str, required=True, help="who promotes it")
    parser.add_argument("--reason", type=str, required=True, help="why, e.g. the shadow results")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.version, args.by, args.reason)

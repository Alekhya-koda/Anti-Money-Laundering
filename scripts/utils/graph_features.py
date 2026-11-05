import os
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd


def process_graph_features_table(
    snapshot_date_str, silver_directory, gold_graph_feature_directory, lookback_days=4
):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # same 4-day window and alert universe as the feature store
    columns = ["src_acct", "dst_acct", "is_self_transfer", "is_cross_currency"]
    pdfs = []
    for i in range(lookback_days, 0, -1):
        day_str = (snapshot_date - timedelta(days=i)).strftime("%Y_%m_%d")
        filepath = silver_directory + "silver_transactions_" + day_str + ".parquet"
        if not os.path.exists(filepath):
            print("not enough history, skipped:", filepath)
            return None
        pdfs.append(pd.read_parquet(filepath, columns=columns))
    df = pd.concat(pdfs, ignore_index=True)
    df = df[~df["is_self_transfer"] & ~df["is_cross_currency"]]

    # one edge per (sender, receiver), weight = number of payments
    edges = df.groupby(["src_acct", "dst_acct"]).size().reset_index(name="n_txn")
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges.itertuples(index=False), weight="n_txn")
    print("graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())

    # feature: pagerank, high when an account receives from many accounts that are also central
    pagerank = nx.pagerank(G, weight="n_txn")

    # feature: size of the strongly connected component, > 1 means the account is in a payment loop
    scc_size = {}
    for component in nx.strongly_connected_components(G):
        for acct in component:
            scc_size[acct] = len(component)

    # feature: 2-hop fan out, how many accounts my receivers pay further
    out_deg = edges.groupby("src_acct")["dst_acct"].nunique()
    edges["dst_out_deg"] = edges["dst_acct"].map(out_deg).fillna(0)
    nbr_out = edges.groupby("src_acct")["dst_out_deg"].agg(["sum", "mean"])
    nbr_out.columns = ["nbr_out_deg_sum", "nbr_out_deg_avg"]

    # feature: 2-hop fan in, how many accounts paid my senders before
    in_deg = edges.groupby("dst_acct")["src_acct"].nunique()
    edges["src_in_deg"] = edges["src_acct"].map(in_deg).fillna(0)
    nbr_in = edges.groupby("dst_acct")["src_in_deg"].agg(["sum", "mean"])
    nbr_in.columns = ["nbr_in_deg_sum", "nbr_in_deg_avg"]

    graph_pdf = pd.DataFrame({"acct": list(G.nodes())})
    graph_pdf["pagerank"] = graph_pdf["acct"].map(pagerank)
    graph_pdf["scc_size"] = graph_pdf["acct"].map(scc_size)
    graph_pdf = graph_pdf.merge(nbr_out, left_on="acct", right_index=True, how="left")
    graph_pdf = graph_pdf.merge(nbr_in, left_on="acct", right_index=True, how="left")
    graph_pdf = graph_pdf.fillna(0)
    graph_pdf["snapshot_date"] = snapshot_date.date()

    partition_name = "gold_graph_features_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = gold_graph_feature_directory + partition_name
    graph_pdf.to_parquet(filepath, index=False)
    print("saved to:", filepath, "row count:", len(graph_pdf))

    return graph_pdf

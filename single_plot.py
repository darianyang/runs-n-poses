from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotting

data_dir = Path("../")
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

annotated_df = pd.read_csv(data_dir / "annotations.csv")
annotated_df["release_date"] = pd.to_datetime(annotated_df["release_date"])

all_similarity_scores = pd.read_parquet(data_dir / "all_similarity_scores.parquet")

# bust_dfs = {}
# for m in plotting.METHODS:
#     filename = data_dir / "posebusters_results" / f"{m}.csv"
#     if filename.exists():
#         bust_dfs[m] = pd.read_csv(filename)

full_datasets = {}
for method in plotting.METHODS:
    filename = data_dir / "predictions" / f"{method}.csv"
    df = pd.read_csv(filename, low_memory=False)
    keep_columns = [
        "target",
        "ligand_instance_chain",
        "lddt_pli",
        "rmsd",
        "lddt_lp",
        "bb_rmsd",
        "seed",
        "sample",
        "ranking_score",
        "ligand_is_proper",
        "prot_lig_chain_iptm_average_lddt_pli",
        "prot_lig_chain_iptm_min_lddt_pli",
        "prot_lig_chain_iptm_max_lddt_pli",
        "lig_prot_chain_iptm_average_lddt_pli",
        "lig_prot_chain_iptm_min_lddt_pli",
        "lig_prot_chain_iptm_max_lddt_pli",
        "prot_lig_chain_iptm_average_rmsd",
        "prot_lig_chain_iptm_min_rmsd",
        "prot_lig_chain_iptm_max_rmsd",
        "lig_prot_chain_iptm_average_rmsd",
        "lig_prot_chain_iptm_min_rmsd",
        "lig_prot_chain_iptm_max_rmsd",
        "model_ligand_chain_lddt_pli",
        "model_ligand_chain_rmsd",
        "ligand_ccd_code",
        "model_ligand_smiles",
        "pred_pocket_f1",
    ]
    if "seed" not in df.columns:
        df["seed"] = 1
    if "sample" not in df.columns:
        df["sample"] = 1
    if "ranking_score" not in df.columns:
        df["ranking_score"] = 1
    if "lig_prot_chain_iptm_average_rmsd" not in df.columns:
        df["lig_prot_chain_iptm_average_rmsd"] = 1
    if "prot_lig_chain_iptm_average_rmsd" not in df.columns:
        df["prot_lig_chain_iptm_average_rmsd"] = 1
    if "pred_pocket_f1" not in df.columns:
        df["pred_pocket_f1"] = 1
    keep_columns = [c for c in keep_columns if c in df.columns]
    full_datasets[method] = (
        df[keep_columns].rename(columns={"target": "system_id"}).reset_index(drop=True)
    )
    full_datasets[method]["group_key"] = (
        full_datasets[method]["system_id"]
        + "__"
        + full_datasets[method]["ligand_instance_chain"]
    )
    full_datasets[method]["method"] = method
    full_datasets[method] = (
        full_datasets[method]
        .sort_values(by=["lddt_pli", "rmsd"], ascending=[False, True])
        .groupby(["group_key", "seed", "sample"])
        .head(1)
        .reset_index(drop=True)
    )
    # if method in bust_dfs:
    #     full_datasets[method] = full_datasets[method].merge(
    #         bust_dfs[method][["system_id", "ligand_instance_chain", "pb_success"]],
    #         on=["system_id", "ligand_instance_chain"],
    #         how="left",
    #     )
    #     full_datasets[method]["pb_success"] = (
    #         full_datasets[method]["pb_success"].fillna(False).astype(float)
    #     )
    # else:
    #     full_datasets[method]["pb_success"] = -1
    full_datasets[method]["pb_success"] = -1

def pivot_df(df, annotated_df):
    df = df.pivot(
        index=[
            "group_key",
            "system_id",
            "ligand_is_proper",
            "ligand_instance_chain",
        ],
        columns="method",
        values=[
            "lddt_pli",
            "rmsd",
            "lddt_lp",
            "bb_rmsd",
            "pb_success",
            "pred_pocket_f1",
        ],
    ).reset_index()
    df.columns = [f"{col[0]}_{col[1]}" if len(col[1]) else col[0] for col in df.columns]
    df = df[df["ligand_is_proper"].fillna(False)].reset_index(drop=True)
    merge_columns = [col for col in annotated_df.columns if col not in df.columns]
    df = df.merge(
        annotated_df[["group_key"] + merge_columns], on="group_key", how="left"
    )
    return df


top_dfs = {}
top_5_dfs = {}
best_5_dfs = {}
best_dfs = {}
worst_dfs = {}
random_dfs = {}
random_5_dfs = {}
rank_by = "lig_prot_chain_iptm_average_rmsd"
for m in full_datasets:
    top_dfs[m] = (
        full_datasets[m]
        .sort_values(by=rank_by, ascending=False)
        .groupby(["system_id", "ligand_instance_chain"])
        .head(1)
    )
    best_dfs[m] = (
        full_datasets[m]
        .sort_values(by="lddt_pli", ascending=False)
        .groupby(["system_id", "ligand_instance_chain"])
        .head(1)
    )
    worst_dfs[m] = (
        full_datasets[m]
        .sort_values(by="lddt_pli", ascending=True)
        .groupby(["system_id", "ligand_instance_chain"])
        .head(1)
    )
    random_dfs[m] = (
        full_datasets[m]
        .sample(frac=1)
        .groupby(["system_id", "ligand_instance_chain"])
        .head(1)
    )
    all_top = (
        full_datasets[m]
        .sort_values(by=rank_by, ascending=False)
        .groupby(["group_key", "seed"])
        .head(1)
    )

    all_top["Rank"] = all_top.groupby("group_key")[rank_by].rank(
        ascending=False, method="first"
    )
    top_5_dfs[m] = []

    for rank in range(1, 6):
        df = all_top[all_top["Rank"] == rank]
        top_5_dfs[m].append(df)

    all_best = (
        full_datasets[m]
        .sort_values(by="lddt_pli", ascending=False)
        .groupby(["group_key", "seed"])
        .head(1)
    )
    all_best["Rank"] = all_best.groupby("group_key")["lddt_pli"].rank(
        ascending=False, method="first"
    )
    best_5_dfs[m] = []
    for rank in range(1, 6):
        df = all_best[all_best["Rank"] == rank]
        best_5_dfs[m].append(df)

    all_random = full_datasets[m].sample(frac=1).groupby(["group_key", "seed"]).head(1)
    all_random["Rank"] = all_random.groupby("group_key")[rank_by].rank(
        ascending=False, method="first"
    )
    random_5_dfs[m] = []
    for rank in range(1, 6):
        df = all_random[all_random["Rank"] == rank]
        random_5_dfs[m].append(df)

results_df_top = pivot_df(pd.concat(top_dfs.values()), annotated_df)
results_df_best = pivot_df(pd.concat(best_dfs.values()), annotated_df)
results_df_worst = pivot_df(pd.concat(worst_dfs.values()), annotated_df)
results_df_random = pivot_df(pd.concat(random_dfs.values()), annotated_df)
results_df_top_5 = []
results_df_best_5 = []
results_df_random_5 = []
for i in range(5):
    results_df_top_5.append(
        pivot_df(pd.concat([top_5_dfs[m][i] for m in top_5_dfs]), annotated_df)
    )
    results_df_best_5.append(
        pivot_df(pd.concat([best_5_dfs[m][i] for m in best_5_dfs]), annotated_df)
    )
    results_df_random_5.append(
        pivot_df(pd.concat([random_5_dfs[m][i] for m in random_5_dfs]), annotated_df)
    )

dfs = {
    "top": results_df_top,
    "best": results_df_best,
    "worst": results_df_worst,
    "random": results_df_random,
}
for i in range(5):
    dfs[f"top_5_{i + 1}"] = results_df_top_5[i]
    dfs[f"best_5_{i + 1}"] = results_df_best_5[i]
    dfs[f"random_5_{i + 1}"] = results_df_random_5[i]

for df_name in dfs:
    dfs[df_name]["lddt_pli_max"] = np.nanmax(
        dfs[df_name][
            [
                f"lddt_pli_{m}"
                for m in plotting.METHODS
                if f"lddt_pli_{m}" in dfs[df_name].columns
            ]
        ],
        axis=1,
    )
    dfs[df_name]["rmsd_min"] = np.nanmin(
        dfs[df_name][
            [
                f"rmsd_{m}"
                for m in plotting.METHODS
                if f"rmsd_{m}" in dfs[df_name].columns
            ]
        ],
        axis=1,
    )
    dfs[df_name]["lddt_pli_average"] = np.nanmedian(
        dfs[df_name][
            [
                f"lddt_pli_{m}"
                for m in plotting.METHODS
                if f"lddt_pli_{m}" in dfs[df_name].columns
            ]
        ],
        axis=1,
    )
    dfs[df_name]["rmsd_average"] = np.nanmedian(
        dfs[df_name][
            [
                f"rmsd_{m}"
                for m in plotting.METHODS
                if f"rmsd_{m}" in dfs[df_name].columns
            ]
        ],
        axis=1,
    )

common_subset_dfs_all = {}
cluster_dfs_all = {}
for df in dfs:
    common_subset_dfs_all[df] = (
        dfs[df]
        .dropna(
            subset=[f"lddt_pli_{method}" for method in plotting.COMMON_SUBSET_METHODS]
            + ["sucos_shape"]
        )
        .reset_index(drop=True)
    )
    cluster_dfs_all[df] = (
        common_subset_dfs_all[df][
            common_subset_dfs_all[df]["ligand_is_proper"]
            & (common_subset_dfs_all[df]["sucos_shape"].notna())
        ]
        .sort_values(by=plotting.SIMILARITY_METRIC)
        .groupby("cluster")
        .head(1)
    )

# make single plot
fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

plotting.make_one_plot(
        common_subset_dfs_all["top"],
        ax[0],
        title=None,
        similarity_metric=plotting.SIMILARITY_METRIC,
        similarity_bins=plotting.SIMILARITY_BINS,
        lddt_pli_threshold=plotting.LDDT_PLI_THRESHOLD,
        rmsd_threshold=plotting.RMSD_THRESHOLD,
        methods=plotting.METHODS,
        legend_loc="upper left",
    )

# plotting.make_one_plot(
#         common_subset_dfs_all["top"],
#         ax[0],
#         title=None,
#         similarity_metric=plotting.SIMILARITY_METRIC,
#         similarity_bins=plotting.SIMILARITY_BINS,
#         lddt_pli_threshold=0, # thus RMSD only ranking
#         rmsd_threshold=3,
#         methods=plotting.METHODS,
#         legend_loc=None,
#         ylabel=f"RMSD < 3 Å Success Rate (%)",
#     )

plotting.make_one_plot(
        common_subset_dfs_all["top"],
        ax[1],
        title=None,
        similarity_metric=plotting.SIMILARITY_METRIC,
        similarity_bins=plotting.SIMILARITY_BINS,
        lddt_pli_threshold=0, # thus RMSD only ranking
        rmsd_threshold=plotting.RMSD_THRESHOLD,
        methods=plotting.METHODS,
        legend_loc=None,
        ylabel=f"RMSD < {plotting.RMSD_THRESHOLD} Å Success Rate (%)",
    )

plt.tight_layout()
plt.savefig(figures_dir / f"single_plot.pdf")

# dist plot
fig, ax = plt.subplots(nrows=2, figsize=(10, 4))
distribution_axes = [ax[0], ax[1]]
accuracy_metrics = ["lddt_pli", "rmsd"]
thresholds = [plotting.LDDT_PLI_THRESHOLD, plotting.RMSD_THRESHOLD]
logs = [False, True]
add_xlabels = [False, True]
for i, (ax, accuracy_metric, threshold, log, add_xlabel) in enumerate(
    zip(distribution_axes, accuracy_metrics, thresholds, logs, add_xlabels)
):
    plotting.make_distribution_plot(
        common_subset_dfs_all["top"],
        ax,
        accuracy_metric,
        threshold,
        similarity_metric=plotting.SIMILARITY_METRIC,
        similarity_bins=plotting.SIMILARITY_BINS,
        methods=plotting.METHODS,
        log=log,
        scatter_spacing=0.2,
        add_xlabel=add_xlabel,
    )

plt.tight_layout()
plt.savefig(figures_dir / f"dist_plot.pdf")
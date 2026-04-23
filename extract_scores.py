import os
import re
import pandas as pd
import json
import string
from multiprocessing import Pool
import yaml
from pathlib import Path
from functools import partial
import numpy as np

def process_boltz_file(json_file, input_data, boltz_output_dir, boltz_analysis_dir):
    pattern = r"^(.*?)_(\d+)_(\d+)\.json$"
    match = re.match(pattern, json_file)
    if not match:
        return [], []

    target_id = match.group(1)
    seed = str(match.group(2))
    sample = str(match.group(3))

    if target_id not in input_data.keys():
        return [], []
    #print(f"Processing file {json_file} for target {target_id}, seed {seed}, sample {sample}...")

    seed_dir = os.path.join(boltz_output_dir, target_id, seed, f"boltz_results_{target_id}" , "predictions", f"{target_id}")

    conf_json = os.path.join(seed_dir, f"confidence_{target_id}_model_{sample}.json")
    
    json_path = os.path.join(boltz_analysis_dir, json_file)

    #print(f"CONF_JSON: {conf_json}, JSON_PATH: {json_path}")
    # check if json_path and conf_json exist
    # if not, then return empty lists, skipping this file for now
    if not os.path.exists(json_path) or not os.path.exists(conf_json):
        #print(f"Warning: Missing files for target {target_id}, seed {seed}, sample {sample}. Skipping.")
        #print(f"MISSING: CONF_JSON: {conf_json}, JSON_PATH: {json_path}")
        return [], []

    try:
        with open(json_path) as f:
            result = json.load(f)
            #print(f"Loaded result from {json_path}: {result.keys()}")
        with open(conf_json) as f:
            data = json.load(f)
            #print(f"Loaded confidence data from {conf_json}: {data.keys()}")
    except Exception as e:
        print(f"Error loading JSON files for target {target_id}, seed {seed}, sample {sample}: {e}")
        return [], []

    try:
        lddt_pli_list = result["lddt_pli"]["assigned_scores"]
        rmsd_list = result["rmsd"]["assigned_scores"]
    except KeyError as e:
        return [], []
    
    if not lddt_pli_list or not rmsd_list:
        return [], []

    boltz_data_lddt_pli = []
    boltz_data_rmsd = []
    
    for item in lddt_pli_list:
        mdl_lig_chain = item["model_ligand"].split(".")[0]
        global_index = string.ascii_uppercase.index(mdl_lig_chain)
        num_prot_chains = len(input_data[target_id]["sequences"])
        ligand_index = global_index - num_prot_chains
        trg_lig_chain = item["reference_ligand"].split("/")[-1].split(".sdf")[0]
        lddt_pli = item["score"]
        lig_prot_pair_iptm = [data["pair_chains_iptm"][str(global_index)][str(idx)] for idx in range(num_prot_chains)]
        lig_prot_pair_iptm_average = sum(lig_prot_pair_iptm) / len(lig_prot_pair_iptm)
        prot_lig_pair_iptm = [data["pair_chains_iptm"][str(idx)][str(global_index)] for idx in range(num_prot_chains)]
        prot_lig_pair_iptm_average = sum(prot_lig_pair_iptm) / len(prot_lig_pair_iptm)
        boltz_data_lddt_pli.append({
            "target": target_id,
            "method": "boltz",
            "seed": seed,
            "sample": sample,
            "ranking_score": data["confidence_score"],
            "prot_lig_chain_iptm_average": prot_lig_pair_iptm_average,
            "prot_lig_chain_iptm_min": min(prot_lig_pair_iptm),
            "prot_lig_chain_iptm_max": max(prot_lig_pair_iptm),
            "lig_prot_chain_iptm_average": lig_prot_pair_iptm_average,
            "lig_prot_chain_iptm_min": min(lig_prot_pair_iptm),
            "lig_prot_chain_iptm_max": max(lig_prot_pair_iptm),
            "lddt_pli": lddt_pli,
            "model_ligand_chain": mdl_lig_chain,
            "model_ligand_ccd_code": input_data[target_id]["ccd_codes"][ligand_index],
            "model_ligand_smiles": input_data[target_id]["smiles"][ligand_index],
            "target_ligand_chain": trg_lig_chain
        })     
        
    for item in rmsd_list:
        mdl_lig_chain = item["model_ligand"].split(".")[0]
        global_index = string.ascii_uppercase.index(mdl_lig_chain)
        num_prot_chains = len(input_data[target_id]["sequences"])
        ligand_index = global_index - num_prot_chains
        trg_lig_chain = item["reference_ligand"].split("/")[-1].split(".sdf")[0]
        rmsd = item["score"]
        lddt_lp = item["lddt_lp"]
        bb_rmsd = item["bb_rmsd"]
        lig_prot_pair_iptm = [data["pair_chains_iptm"][str(global_index)][str(idx)] for idx in range(num_prot_chains)]
        lig_prot_pair_iptm_average = sum(lig_prot_pair_iptm) / len(lig_prot_pair_iptm)
        prot_lig_pair_iptm = [data["pair_chains_iptm"][str(idx)][str(global_index)] for idx in range(num_prot_chains)]
        prot_lig_pair_iptm_average = sum(prot_lig_pair_iptm) / len(prot_lig_pair_iptm)
        boltz_data_rmsd.append({
            "target": target_id,
            "method": "boltz",
            "seed": seed,
            "sample": sample,
            "ranking_score": data["confidence_score"],
            "prot_lig_chain_iptm_average": prot_lig_pair_iptm_average,
            "prot_lig_chain_iptm_min": min(prot_lig_pair_iptm),
            "prot_lig_chain_iptm_max": max(prot_lig_pair_iptm),
            "lig_prot_chain_iptm_average": lig_prot_pair_iptm_average,
            "lig_prot_chain_iptm_min": min(lig_prot_pair_iptm),
            "lig_prot_chain_iptm_max": max(lig_prot_pair_iptm),
            "rmsd": rmsd,
            "lddt_lp": lddt_lp,
            "bb_rmsd": bb_rmsd,
            "model_ligand_chain": mdl_lig_chain,
            "model_ligand_ccd_code": input_data[target_id]["ccd_codes"][ligand_index],
            "model_ligand_smiles": input_data[target_id]["smiles"][ligand_index],
            "target_ligand_chain": trg_lig_chain
        })

    return boltz_data_lddt_pli, boltz_data_rmsd

def merge_to_final_df(lddt_pli, rmsd, ref_df):
    df_lddt_pli = pd.DataFrame(lddt_pli)
    df_rmsd = pd.DataFrame(rmsd)
    
    #print(f"df_lddt_pli: {df_lddt_pli}")
    print(f"df_lddt_pli columns: {df_lddt_pli.columns}")
    #print(f"df_rmsd: {df_rmsd}")
    print(f"df_rmsd columns: {df_rmsd.columns}")

    df_lddt_pli_name = (
        pd.merge(
            df_lddt_pli,
            ref_df[["system_id", "ligand_instance_chain", "ligand_ccd_code"]],
            how="left",
            left_on=["target", "target_ligand_chain"],
            right_on=["system_id", "ligand_instance_chain"]
        ).drop(columns=["system_id", "ligand_instance_chain"])
    )
    
    df_rmsd_name = (
        pd.merge(
            df_rmsd,
            ref_df[["system_id", "ligand_instance_chain", "ligand_ccd_code"]],
            how="left",
            left_on=["target", "target_ligand_chain"],
            right_on=["system_id", "ligand_instance_chain"]
        ).drop(columns=["system_id", "ligand_instance_chain"])
    )

    df_lddt_rmsd = pd.merge(
        df_lddt_pli_name,
        df_rmsd_name,
        how="outer",
        on=["target", "method", "seed", "sample", "ranking_score", "model_ligand_ccd_code", "model_ligand_smiles", "ligand_ccd_code"],
        indicator=True,          
        suffixes=("_lddt_pli", "_rmsd")
    )

    df_final = pd.merge(
        df_lddt_rmsd,
        ref_df[["system_id", "ligand_ccd_code", "ligand_instance_chain", "ligand_is_proper"]],
        how="left",    
        left_on=["target", "model_ligand_ccd_code", "target_ligand_chain_lddt_pli"],
        right_on=["system_id", "ligand_ccd_code", "ligand_instance_chain"]
    )

    df_final.drop(["system_id", "_merge", "target_ligand_chain_rmsd", "target_ligand_chain_lddt_pli"], axis=1, inplace=True)
    df_final = df_final.drop_duplicates()

    return df_final

ref_df = pd.read_csv("../annotations.csv")
input_json = "../inputs.json"
with open(input_json, 'r') as f:
    input_data = json.load(f)

METHODS = ["boltz-r2"]

dfs = pd.DataFrame()

for method in METHODS:
    analysis_dir = f"{method}/analysis"
    out_dir = f"{method}/outputs"

    lddt_pli = list()
    rmsd=list()
    filenames = list(os.listdir(analysis_dir))

    if method == "boltz" or method == "boltz-r0" or method == "boltz-r2":
        print(f"Processing method {method} with multiprocessing...")
        partial_func = partial(process_boltz_file, input_data=input_data, 
                               boltz_output_dir=out_dir, boltz_analysis_dir=analysis_dir)
        with Pool(processes=128) as pool:
            for lddt_pli_single, rmsd_single in pool.imap(partial_func, filenames):
                lddt_pli.extend(lddt_pli_single)
                rmsd.extend(rmsd_single)

        # for serial testing
        #stop_index = 100
        # for json_file in filenames:
        #     lddt_pli_single, rmsd_single = process_boltz_file(json_file, input_data, out_dir, analysis_dir)
        #     lddt_pli.extend(lddt_pli_single)
        #     rmsd.extend(rmsd_single)
        #     # if len(lddt_pli) >= stop_index:
        #     #     #print(f"{lddt_pli_single} {rmsd_single}")
        #     #     break # for serial test

    df = merge_to_final_df(lddt_pli, rmsd, ref_df)
    
    dfs = pd.concat([dfs, df], ignore_index=True)

# save the final dataframe to a csv file
dfs.to_csv("final_scores_boltz-r2.csv", index=False)

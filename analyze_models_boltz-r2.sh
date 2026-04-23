#!/bin/bash

n_cpus=384
boltz="boltz-r2"

#models=$(find "boltz-r0/outputs" -type f -name "input_model_*.cif")
models=$(find "$boltz/outputs" -type f -name "*.cif")
#models=$(find "boltz-r0/outputs/9f41__1__1.A__1.Q" -type f -name "*.cif")

# count total models
total_models=$(echo "$models" | wc -l)
echo "Total models to analyze: $total_models"

# ignore networkx runtime warnings
export PYTHONWARNINGS="ignore::RuntimeWarning"

# make analysis dir
mkdir -p $boltz/analysis

# loop over all models and allocate CPUs for parallel processing
model_index=0
cpu_index=0
for line in $models; do
    echo "CPU $cpu_index : analyzing model $((model_index + 1)) / $total_models"
    model_index=$((model_index + 1))

    # define vars from line
    target_id=$(echo "$line" | cut -d'/' -f3)
    seed=$(echo "$line" | cut -d'/' -f4)
    model_id_cif=$(basename "$line")
    # model_id is the -4th char of model_if_cif string
    # remove up to last _ and then remove .cif
    model_id=${model_id_cif##*_}
    model_id=${model_id%%.*}
    sdf_files=("../ground_truth/$target_id/ligand_files/"*.sdf)

    echo "Processing target $target_id, seed $seed, model $model_id"

    # if "boltz-r0/analysis/${target_id}_${seed}_${model_id}.json" already exists, skip
    if [ -f "$boltz/analysis/${target_id}_${seed}_${model_id}.json" ]; then
        echo "...analysis for target $target_id, seed $seed, model $model_id already exists. Skipping."
        continue
    fi

    # run ost compare-ligand-structures in background for current model
    ost compare-ligand-structures \
        -m "$line" \
        -rl "${sdf_files[@]}" \
        -r "../ground_truth/$target_id/receptor.cif" \
        -o "$boltz/analysis/${target_id}_${seed}_${model_id}.json" \
        --lddt-pli --rmsd --lddt-pli-amc &

    # increment cpu index and check if we need to wait for current batch to finish
    cpu_index=$((cpu_index + 1))
    if [ $cpu_index -ge $n_cpus ]; then
        echo "Waiting for current batch of $n_cpus analyses at model $model_index to finish..."
        wait # wait for all background processes to finish before starting next batch
        cpu_index=0
    fi
done
# wait for any remaining background processes to finish
# e.g. if total_models is not divisible by n_cpus
echo "Waiting for final batch of analyses to finish..."
wait

# # serial run
# model_index=0
# for line in $models; do
#     echo "Analyzing model $((model_index + 1)) / $total_models"
#     model_index=$((model_index + 1))

#     # define vars from line
#     target_id=$(echo "$line" | cut -d'/' -f3)
#     seed=$(echo "$line" | cut -d'/' -f4)
#     model_id_cif=$(basename "$line")
#     # model_id is the -4th char of model_if_cif string
#     # remove up to last _ and then remove .cif
#     model_id=${model_id_cif##*_}
#     model_id=${model_id%%.*}
#     #model_id=$(echo "$model_id_cif" | cut -d'_' -f11 | cut -d'.' -f1)
#     sdf_files=("../ground_truth/$target_id/ligand_files/"*.sdf)

#     echo "Processing target $target_id, seed $seed, model $model_id"
#     echo "model_id_cif $model_id_cif"
#     echo "file $line"

#     # # if "boltz-r0/analysis/${target_id}_${seed}_${model_id}.json" already exists, skip
#     if [ -f "boltz-r0/analysis/${target_id}_${seed}_${model_id}.json" ]; then
#         echo "...analysis for target $target_id, seed $seed, model $model_id already exists. Skipping."
#         continue
#     fi

#     # ost compare-ligand-structures \
#     #     -m "$line" \
#     #     -rl "${sdf_files[@]}" \
#     #     -r "../ground_truth/$target_id/receptor.cif" \
#     #     -o "boltz-r0/analysis/${target_id}_${seed}_${model_id}.json" \
#     #     --lddt-pli --rmsd --lddt-pli-amc
#     #break # only do one for testing
# done

#!/bin/bash

# total of 2594 input files (should have been 2600 but 6 had errors in input gen)
# divide into n chunks and use n GPUs to run them in parallel
# first generate a list of all input files
n_gpus=4
input_files=$(ls boltz-r0/inputs/*.yaml)
total_files=$(echo "$input_files" | wc -l)
 # divide into n_gpus chunks, rounding up
files_per_gpu=$(( (total_files + 3) / n_gpus ))
echo "Total files: $total_files"
echo "Files per GPU: $files_per_gpu"

# for specified GPU: run the corresponding chunk of files
# it needs to run serially for each GPU
# but then the n GPUs will run in parallel
gpu=$1
# ensure that gpu index is included
if [ -z "$gpu" ]; then
  echo "Usage: $0 <gpu_index>"
  echo "Error: Enter a valid GPU index from 0 to $((n_gpus - 1))"
  exit 1
fi

# run predictions for current chunk
start_index=$(( gpu * files_per_gpu + 1 ))
end_index=$(( (gpu + 1) * files_per_gpu ))
chunk_files=$(echo "$input_files" | sed -n "${start_index},${end_index}p")
echo "GPU $gpu: Processing files $start_index to $end_index"
for file in $chunk_files; do
  # run 5 seeds for each input file
  for i in {1..5}; do
    seed=$(od -An -N4 -t u4 /dev/urandom | tr -d ' ')
    CUDA_VISIBLE_DEVICES=$gpu boltz predict \
      --out_dir "boltz-r0/outputs/$(basename "$file" .yaml)/$seed" \
      "$file" \
      --recycling_steps 10 \
      --diffusion_samples 5 \
      --checkpoint "r0-e30_conf-e2.ckpt" \
      --seed $seed \
      --no_potentials
  done
done

####################################################
### example boltz run command for a single input ###
####################################################
# for i in {1..5}; do
#   seed=$(od -An -N4 -t u4 /dev/urandom | tr -d ' ')
#   CUDA_VISIBLE_DEVICES=1 boltz predict \
#     --out_dir "boltz-r0/outputs/8c3u__1__1.A__1.C/$seed" \
#     "boltz-r0/inputs/8c3u__1__1.A__1.C.yaml" \
#     --recycling_steps 10 \
#     --diffusion_samples 5 \
#     --checkpoint "r0-e30_conf-e2.ckpt" \
#     --seed $seed \
#     --no_potentials
# done
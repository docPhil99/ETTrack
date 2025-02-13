#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=valentwin-simcse-roberta-ft-dstl-n-100-hn-1-only-lr-x-bs-256-l-triplet-20231117

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=a.andito@sussex.ac.uk

# run the application
module purge
module load cuda/11.2
module load python/anaconda3
source $condaDotFile
source activate root
conda activate track

python tools/run_hybrid_sort_dance.py 
-f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py 
-b 1 
-d 1 
--fp16 
--fuse 
--expn hxd








#export WANDB_PROJECT=valentwin_dstl
#export WANDB_MODE=offline

#python main_json.py
#python llama.py
#python test.py
#    --model_name_or_path roberta-base \
#    --train_file ../../data/dstl/contrastive/100-hn-1/train.csv \
#    --validation_file ../../data/dstl/contrastive/100-hn-1/val.csv \
#    --eval_file ../../data/dstl/contrastive/100-hn-1/test.csv \
#    --output_dir ../result/valentwin-simcse-roberta-ft-dstl-n-100-hn-1-only-lr-5e5-bs-256-l-triplet-20231117 \
#    --num_train_epochs 3 \
#    --per_device_train_batch_size 256 \
#    --per_device_eval_batch_size 256 \
#    --learning_rate 5e-5 \
#    --max_seq_length 32 \
#    --pooler_type cls \
#    --loss_function TripletMarginWithDistanceLoss \
#    --overwrite_output_dir \
#    --temp 0.05 \
#    --do_train \
#    --do_eval \
#    --label_names [] \
#    --logging_strategy epoch \
#    --evaluation_strategy epoch \
#    --save_strategy epoch \
#    --metric_for_best_model accuracy \
#    --load_best_model_at_end \
#    --report_to wandb \
#    --run_name valentwin-simcse-roberta-ft-dstl-n-100-hn-1-only-lr-5e5-bs-256-l-triplet-20231117 \
#    --fp16 \
#    "$@"

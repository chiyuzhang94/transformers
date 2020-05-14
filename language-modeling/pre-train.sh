#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --job-name=roberta
#SBATCH --output=roberta.out
#SBATCH --account=rrg-mageed
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
module load cuda cudnn
source ~/roberta/bin/activate


python3 run_language_modeling.py \
	--gradient_accumulation_steps 32 \
	--train_data_file ./data/oscar.eo.txt \
	--target_dictionary_file ./sample_config/given_dict.json \
	--output_dir ./sample_model/ \
	--model_type roberta \
	--mlm \
	--config_name ./sample_config \
	--tokenizer_name ./sample_config \
	--do_train \
	--line_by_line \
	--learning_rate 1e-4 \
	--num_train_epochs 10 \
	--save_total_limit 2 \
	--save_steps 476 \
	--per_gpu_train_batch_size 8 \
	--seed 42

#!/bin/bash

source ~/roberta/bin/activate
/bin/hostname -s
echo "PROCID: $SLURM_PROCID"
echo "LOCALID: $SLURM_LOCALID"
source ~/roberta/bin/activate

python3 -m torch.distributed.launch \
		--nproc_per_node=$NPROC_PER_NODE \
		--nnodes=$SLURM_JOB_NUM_NODES \
		--node_rank=$SLURM_PROCID \
		--master_addr="$PARENT" --master_port="$MPORT" \
		run_language_modeling_V2.py \
		--gradient_accumulation_steps 16 \
		--train_data_file ./data/sample.txt \
		--target_dictionary_file ./sample_config/given_dict.json \
		--output_dir ./sample_model/ \
		--model_type roberta \
		--mlm \
		--local_rank $SLURM_PROCID \
		--config_name ./sample_config \
		--tokenizer_name ./sample_config \
		--do_train \
		--line_by_line \
		--learning_rate 1e-4 \
		--num_train_epochs 40 \
		--save_total_limit 5 \
		--save_steps 20 \
		--per_gpu_train_batch_size 16 \
		--seed 42
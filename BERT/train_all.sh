#!/usr/bin/env bash

base_model_source=/lium/home/vpelloin/git/bert-for-slu/source_models/models/HuggingFace_format
model_types=(
'flaubert/flaubert_base_cased'
'flaubert/flaubert_base_uncased'
# 'flaubert_base_uncased_xlm'
'flaubert_base_uncased_xlm_fine_tuned_204'
'flaubert_base_uncased_xlm_mixed'
'flaubert_base_uncased_xlm_only_asr_63'
'flaubert_base_uncased_xlm_only_asr_63_newbpe'
)

date=$(date +"%d-%m-%y_%H-%M")

max_length=256
batch_size=32
num_epochs=100
save_steps=1500
learning_rate=5e-5
weight_decay=0.0
seed=1

data_dir=../Data/MEDIA_inv

. parse_options.sh

base_output_dir=FineTune_FlauBERT/$date

# exporting list of labels
cat $data_dir"/train.txt" $data_dir"/dev.txt" $data_dir"/test.txt" |
	cut -d " " -f 2 |
	grep -v "^$" |
	sort |
	uniq > $data_dir"/labels.txt"

# training each model
for model_type in "${model_types[@]}"; do
	echo $model_type

	output_dir=$base_output_dir/$model_type
	cache_dir=$output_dir/cache
	mkdir -p $output_dir
	mkdir -p $cache_dir

	other_options=
	lower=
	if [ "$model_type" = "flaubert/flaubert_base_cased" ]; then
		model_source=$model_type
	elif [ "$model_type" = "flaubert/flaubert_base_uncased" ]; then
		model_source=$model_type
		other_options='--do_lower_case'
		lower=_lower
	else
		model_source=$base_model_source/$model_type
		other_options='--do_lower_case'
		lower=_lower
	fi

	mkdir -p $output_dir/data
	cp -r $data_dir$lower/*.txt $output_dir/data/.

	# python3 run_slu.py \
	# 	--data_dir $output_dir/data \
	# 	--labels $output_dir/data/labels.txt \
	# 	--model_type flaubert \
	# 	--model_name_or_path $model_source \
	# 	--output_dir $output_dir \
	# 	--max_seq_length $max_length \
	# 	--num_train_epochs $num_epochs \
	# 	--per_gpu_train_batch_size $batch_size \
	# 	--save_steps $save_steps \
	# 	--seed $seed \
	# 	--learning_rate $learning_rate \
	# 	--weight_decay $weight_decay \
	# 	--cache_dir $cache_dir \
	# 	--evaluate_during_training \
	# 	--keep_accents \
	# 	--overwrite_output_dir \
	# 	--do_train \
	# 	--do_eval \
	# 	--do_predict \
	# 	--eval_all_checkpoints \
	# 	$other_options |
	# 	tee $output_dir/train.log

	# obtaining best dev model, and saving it with a symbolic link
	rm -f $output_dir/checkpoint-best
	best_checkpoint=$(
		cat $output_dir/dev_results.txt |
			grep '_cer = ' |
			LC_ALL=C sort -r -g -t ' ' -k 3,3 |
			tail -n 1 |
			sed -r 's/^([0-9]+)_cer = .*$/\1/g'
	)
	echo "Best checkpoint for $model_type = $best_checkpoint"
	ln -s checkpoint-$best_checkpoint $output_dir/checkpoint-best

	echo -e "\n\n\n-----------------\n\n\n"
done


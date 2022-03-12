#!/usr/bin/env bash

set -e

model_types=(
'flaubert/flaubert_base_cased'
'flaubert/flaubert_base_uncased'
'flaubert_base_uncased_xlm'
'flaubert_base_uncased_xlm_fine_tuned_204'
'flaubert_base_uncased_xlm_mixed'
'flaubert_base_uncased_xlm_only_asr_63'
'flaubert_base_uncased_xlm_only_asr_63_newbpe'
)

date=$(date +"%d-%m-%y_%H-%M")

max_length=256
batch_size=32
seed=1

data_name=asr-ref
data_dir=../Data/MEDIA_inv

# list of output labels
labels=../Data/MEDIA_inv/labels.txt

# mapping file to convert HuggingFace uttids to Kaldi uttids (used in evaluation)
id_mapping=/export/home/lium/vpelloin/lium_corpus_base/MEDIA/slu/id_mapping_luna2kaldi.txt

checkpoint_name=checkpoint-best

. parse_options.sh

base_output_dir=FineTune_FlauBERT/$date

# training each model
for model_type in "${model_types[@]}"; do
	echo "$model_type/$checkpoint_name"

	model_path=$base_output_dir/$model_type/$checkpoint_name
	output_dir=$base_output_dir/$model_type/$checkpoint_name/$data_name

	other_options=
	lower=
	if [ ! "$model_type" = "flaubert/flaubert_base_cased" ]; then
		other_options='--do_lower_case'
		if [ "$data_dir" = "../Data/MEDIA_inv" ]; then
			lower=_lower
		else
			echo "WARNING: not using lowercased version of dataset as it has been changed!"
		fi
	fi

	# the run_slu.py script needs the model files to be present in the
	# --output_dir folder, but we want different output folders for different
	# input data samples, so we create specific output dirs, and link the
	# model in this folder.
	mkdir -p $output_dir
	model_files=$(find $model_path/ -maxdepth 1 -type f -printf "%f\n")
	for model_file in $model_files; do
		ln -sf $model_path/$model_file $output_dir/$model_file
	done

	# predicting
	python3 run_slu.py \
		--data_dir $data_dir$lower \
		--labels $labels \
		--model_type flaubert \
		--model_name_or_path $model_path \
		--output_dir $output_dir \
		--max_seq_length $max_length \
		--per_gpu_train_batch_size $batch_size \
		--seed $seed \
		--overwrite_output_dir \
		--overwrite_cache \
		--keep_accents \
		--do_eval \
		--do_predict \
		$other_options

	# computing concept errors rates (CER, CVER, VER)
	for dataset in validation test; do
		echo
		echo "##########"
		echo "Evaluation of $dataset dataset"
		./evalBIO.py \
			--output-dir $output_dir \
			--subset $dataset \
			--utt-ids-mapping $id_mapping
	done

	echo -e "\n\n\n-----------------\n\n\n"
done


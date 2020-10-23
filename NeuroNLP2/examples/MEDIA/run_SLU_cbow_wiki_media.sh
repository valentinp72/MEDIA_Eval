#!/usr/bin/env bash

working_dir=""
trained="trained_wiki_media"
data_dir=$working_dir"/Data/"$trained
embed="cbow"
train=$data_dir"/"$embed"/cbow_train_wmanuel_lref.u8.txt"
dev=$data_dir"/"$embed"/cbow_dev_wmanuel_lref.u8.txt"
test_=$data_dir"/"$embed"/cbow_test_wmanuel_lref.u8.txt"
res_dir=$working_dir"/results/MEDIA/freeze/"$trained"/"$embed


if [ ! -d $res_dir ]; then
   mkdir -p $res_dir
   mkdir $res_dir"/models"
   mkdir $res_dir"/log"
   mkdir $res_dir"/predictions"
fi


for m in LSTM; do
  for n in {1,2,3}; do
  for b in {16,32,64}; do
    for h in {128,256,512}; do

	 python SLU_BLSTM_freeze.py --mode $m --num_epochs 200 --batch_size $b --hidden_size $h --num_layers $n\
	 --char_dim 30 --num_filters 30 --tag_space 0 \
 	--learning_rate 0.001 --decay_rate 0.05 --schedule 1 --gamma 0.0 --task MEDIA --optim ADAM\
	 --dropout std --p 0.5 --unk_replace 0.0 --bidirectional True\
 	--data_path $res_dir --modelname $model_name  --train $train  --dev $dev  --test $test_
done
done
done
done

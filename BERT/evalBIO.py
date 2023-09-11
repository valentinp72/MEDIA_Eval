#!/usr/bin/env python3

import os
import sys
import logging
import argparse

from datasets import load_dataset

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

def evaluation(args):

    if not args.without_HF_dataset:
        # loading MEDIA
        media_datasets = load_dataset(
            "vpelloin/MEDIA", use_auth_token=True, # private dataset
            gen_task='slu', mode='relax'
        )

    # mapping from Luna (used in HuggingFace dataset) ids to Kaldi (used in evaluation script)
    if args.without_HF_dataset:
        with open(args.utt_ids_mapping, 'r') as f:
            utt_id_list = f.read().splitlines()
    else:
        if args.utt_ids_mapping is not None:
            with open(args.utt_ids_mapping, 'r') as f:
                utt_ids_mapping = {k:v for k,v in [line.split('\t') for line in f.read().splitlines()]}
        else:
            utt_ids_mapping = {}

    # in HuggingFace dataset: validation | in output files: dev
    subset_name = args.subset if args.subset != "validation" else "dev"

    os.makedirs(f"{args.output_dir}/{subset_name}", exist_ok=True)
    with open(f"{args.output_dir}/{subset_name}_predictions.txt", 'r') as f_in, \
         open(f"{args.output_dir}/{subset_name}/tagged_hyp.txt", 'w') as f_out_tagged_hyp, \
         open(f"{args.output_dir}/{subset_name}/tagged_ref.txt", 'w') as f_out_tagged_ref, \
         open(f"{args.output_dir}/{subset_name}/aligned_results.txt", 'w') as f_out_aligned_results:

        lines = f_in.read().splitlines()

        outs = {}
        utterances = {}

        # reading predictions, but we don't have utterance ids: we have to map
        # them with the HuggingFace dataset. However, empty utterances are not
        # in the prediction list, so we take careful steps to keep the alignment
        i = 0

        if args.without_HF_dataset:
            while '' in lines:
                end_idx = lines.index('')
                utt_id = utt_id_list[i]
                utterances[utt_id] = lines[:end_idx]
                lines = lines[end_idx + 1:]
                i += 1
        else:
            while '' in lines:
                end_idx = lines.index('')
                utt_id = media_datasets[args.subset][i]['id']
                if len(media_datasets[args.subset][i]['words']) == 0:
                    utterances[utt_id] = []
                else:
                    utterances[utt_id] = lines[:end_idx]
                    lines = lines[end_idx + 1:]
                i += 1

        # converting utterances from BIO to <concept> subwords </>
        for utt_id in utterances:

            last_label = 'O'
            out = []
            for j, (subword, label) in enumerate([x.split(' ') for x in utterances[utt_id]]):
                if label.startswith('B-'):
                    if last_label != 'O':
                        out.append('</>')
                    out.append(f'<{label[2:]}>')
                    last_label = label
                elif label[2:] != last_label[2:]:
                    if last_label != 'O' and label == 'O':
                        out.append('</>')

                if subword.endswith('</w>'):
                    out.append(subword[:-4])
                else:
                    out.append(subword)

            if last_label != 'O':
                out.append('</>')
            out = " ".join(out).lower()
            outs[utt_id] = out

            ref = ""

            if not args.without_HF_dataset:
                if utt_id in utt_ids_mapping:
                    utt_id = utt_ids_mapping[utt_id]

            # adding these predictions to the output files
            print(f'{utt_id}\t{ref}', file=f_out_tagged_ref)
            print(f'{utt_id}\t{out}', file=f_out_tagged_hyp)

            print(f'{utt_id}',   file=f_out_aligned_results)
            print(f'REF: {ref}', file=f_out_aligned_results)
            print(f'HYP: {out}', file=f_out_aligned_results)
            print(f'STP: ',      file=f_out_aligned_results)
            print(f'WER: 0.00%', file=f_out_aligned_results)
            print(file=f_out_aligned_results)

    # calling the eval_cer_cver script to get CER/CVER/VER values
    os.system(
        f"/export/home/lium/vpelloin/git/espresso/slu/tools/eval_cer_cver " \
        f"--dir {args.output_dir}/{subset_name} "
        f"--results-char2word no --results-clean yes " \
        f"media {subset_name}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir', default=None, required=True, type=str,
        help='Base output directory where evaluation results will be saved. A ' \
        'file `{SUBSET}_predictions.txt` containing predictions is required in this folder.'
    )
    parser.add_argument(
        '--subset', default=None, required=True, type=str,
        help='Name of dataset being evaluated (should be train, validation or ' \
        'test)'
    )
    parser.add_argument(
        '--without_HF_dataset', default=False, action='store_true',
        help='If true, do not use HF dataset to evaluate, but use the ids given ' \
        'in --utt-ids-mapping.'
    )
    parser.add_argument(
        '--utt-ids-mapping', default=None, type=str,
        help='If you want to translate output utterances ids to an other format' \
        ' pass a mapping file here (two columns separated by a tab)'
    )
    args = parser.parse_args()
    evaluation(args)

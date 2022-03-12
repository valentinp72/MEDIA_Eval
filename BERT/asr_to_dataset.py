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

def convert(args):

    # loading MEDIA
    media_datasets = load_dataset(
        "vpelloin/MEDIA", use_auth_token=True, # private dataset
    )

    contents = {}

    hugging_face_mapping = {}
    for i, x in enumerate(media_datasets[args.subset]):
        if args.id:
            key = x['id']
        else:
            key = (x['file_id'], x['start_time'], x['end_time'])
        hugging_face_mapping[key] = i

    with open(args.stm_file, 'r') as f:
        utterances = f.read().splitlines()

        for i, utterance in enumerate(utterances):
            if args.id:
                utt_id, text = utterance.split(' ', maxsplit=1)
                key = utt_id
            else:
                file, speaker, utt_id, start, end, other, text = utterance.split(' ', maxsplit=6)
                file_id = file[1:]
                key = (file_id, start, end)

            assert key in hugging_face_mapping, f"{key} -- {utterance} {i}"
            hugging_face_idx = hugging_face_mapping[key]

            content = text \
                .replace("d' accord", "d'accord") \
                .replace('(%hesitation)', 'euh') \
                .replace('est ce', 'est-ce') \
                .replace('a t il', 'a-t-il') \
                .replace('o. k.', 'ok')
            content = [tok for tok in content.split(' ')]
            contents[hugging_face_idx] = content

    with open(args.output_dataset_file, 'w') as f:
        for i, x in enumerate(media_datasets[args.subset]):
            assert i in contents, f"Segment {x} ({i}) not in contents"
            content = "\n".join(contents[i])
            if len(content) == 0:
                content = 'euh'
            if len(x['words']) > 0 and len(content) > 0:
                print(content, end='\n\n', file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stm-file', default=None, required=True, type=str,
        help='Input stm file'
    )
    parser.add_argument(
        '--id', default=False, action='store_true',
        help='Input STM is already a file containing the HuggingFace id followed by text'
    )
    parser.add_argument(
        '--output-dataset-file', default=None, required=True, type=str,
        help='Output file'
    )
    parser.add_argument(
        '--subset', default=None, required=True, type=str,
        help='Name of dataset being evaluated (should be train, validation or ' \
        'test)'
    )
    args = parser.parse_args()
    convert(args)

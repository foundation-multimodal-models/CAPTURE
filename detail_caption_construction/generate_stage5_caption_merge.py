"""
    Copyright (2024) CAPTURE project Authors 

    Licensed under the Apache License, Version 2.0 (the "License"); 
    you may not use this file except in compliance with the License. 
    You may obtain a copy of the License at 

        http://www.apache.org/licenses/LICENSE-2.0 

    Unless required by applicable law or agreed to in writing, software 
    distributed under the License is distributed on an "AS IS" BASIS, 
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
    See the License for the specific language governing permissions and 
    limitations under the License.
"""


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import pandas as pd
import yaml
import tqdm
import argparse

from detail_caption_construction.utils.utils import get_data_files


def caption_merge(llm, left_padding_tokenizer, batch):
    def get_prompt(overall_caption, local_captions, mode="augment", icl=False):
        local_captions = [(h*w, caption) for [x1, y1, h, w], caption in local_captions]
        local_captions = [item[1].strip('.').strip(',') for item in local_captions]
        prompt = ""
        prompt = prompt + "Overall description: " + overall_caption + "\n\n"
        prompt = prompt + " Elements appearing in the image: " + '. '.join(local_captions) + '. \n\n'
        prompt = prompt + "Please combine them into a description, refining the overall description with detailed annotations. " + \
                    "Avoid simply concatenating the sentences. Ignore elements that do not fit in the scene. Reply with no more than three hundred words.\n\n" + \
                    "Refined description: "
        return prompt

    batch_processed_data = [sample.to_dict() for i, sample in batch.iterrows()]
    if 'filtered_overall_caption' in batch_processed_data[0].keys() and 'filtered_local_caption' in batch_processed_data[0].keys():
        overall_caption_key, local_caption_key = 'filtered_overall_caption', 'filtered_local_caption'
    else:
        overall_caption_key, local_caption_key = 'overall_caption', 'local_caption'

    if left_padding_tokenizer.pad_token == left_padding_tokenizer.unk_token:
        overall_captions = [f"<s>{get_prompt(sample[overall_caption_key], eval(sample[local_caption_key]))} " for sample in batch_processed_data]            
        tokenized_input = left_padding_tokenizer(overall_captions, add_special_tokens=False, padding='longest', return_tensors='pt')
        input_ids, attention_mask = tokenized_input['input_ids'].to(llm.device), tokenized_input['attention_mask'].to(llm.device)

        with torch.inference_mode():
            res = left_padding_tokenizer.batch_decode(llm.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                eos_token_id=left_padding_tokenizer.eos_token_id,
                pad_token_id=left_padding_tokenizer.pad_token_id,
                use_cache=True,
                max_new_tokens=500, 
                do_sample=False, 
                temperature=0.2,
                num_beams=3, 
                top_p=0.95, 
                num_return_sequences=1), skip_special_tokens=True)

        res = [this_res.replace(overall_captions[i].replace('<s>', ''), '').strip('\n ') for i, this_res in enumerate(res)]

    for i, sample in enumerate(batch_processed_data):
        sample['synthesized_caption'] = res[i]

    return  batch_processed_data


def main(config_path, chunk_index, chunk_num, node_index, node_num):
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = f"{config['ckpt_path']}/{config['model_path']}"
    print(f"loading {path}")
    left_padding_tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
    left_padding_tokenizer.pad_token = left_padding_tokenizer.unk_token     
    llm = AutoModelForCausalLM.from_pretrained(path, device_map='auto')
    llm.eval().cuda()

    source_data_files, target_data_files = get_data_files(config, node_index=node_index, node_num=node_num)
    batch_size = config['batch_size']

    for source_data_file in source_data_files:
        if f"{source_data_file.split('/')[-1].split('.')[0]}_processed" in target_data_files:
            print(f"file {source_data_file} processed, skipping")       
            continue 
        print(f"processing {source_data_file}")
        processed_data = []
        df = pd.read_parquet(source_data_file)

        start, end = chunk_index * (len(df) // chunk_num), (chunk_index + 1) * (len(df) // chunk_num) - 1
        if len(df) - end < len(df) // chunk_num:
            end = len(df) - 1
        df = df.loc[start: end]
        for offset in tqdm.trange(0, len(df), batch_size):
            offset += start
            batch = df.loc[offset: offset + batch_size - 1].reset_index(drop=True)
            with torch.no_grad():
                batch_processed_data = caption_merge(llm, left_padding_tokenizer, batch)
            processed_data.extend(batch_processed_data)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        base_path = os.path.basename(source_data_file)
        output_path = f"detail_caption_construction/data/processed_data/{base_path.split('.')[0]}_chunk{chunk_index}.parquet"
        processed_df.to_parquet(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_index', type=int)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=10)
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        chunk_index=args.chunk_index, 
        chunk_num=args.chunk_num,
        node_index=args.node_index,
        node_num=args.node_num,
    )








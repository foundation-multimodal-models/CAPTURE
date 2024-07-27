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

import io
import torch
import pandas as pd
import yaml
import tqdm
import argparse
from PIL import Image

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
from detail_caption_construction.utils.utils import get_data_files


def spotter_llava(model, batch, image_processor, tokenizer, img_key):
    images = [Image.open(io.BytesIO(batch.loc[i, img_key])).convert("RGB") for i in range(len(batch))]
    # from IPython import embed; embed()
    try: 
        image_tensor = process_images(images, image_processor, None).to(model.dtype)
    except Exception as e:
        all_image_tensor = []
        for image in images:
            try:
                this_image_tensor = process_images([image], image_processor, None).to(model.dtype)
                all_image_tensor.append(this_image_tensor)
            except Exception as e:
                print("an image is corrupted, skipping")
        image_tensor = torch.cat(all_image_tensor)
        
    llava_ori_prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n'
    spotter_prompt = ' Describe this image in detail. ASSISTANT:'
    spotter_prompts = [llava_ori_prompt + spotter_prompt for _ in range(image_tensor.shape[0])]
    input_ids = tokenizer_image_token(spotter_prompts[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(model.device)
    all_input_ids = torch.tile(input_ids, [image_tensor.shape[0], 1])
    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                all_input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=192,
                use_cache=True)

    res = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    batch_processed_data = []
    for i in range(len(res)):
        processed_sample = batch.loc[i].to_dict()
        res[i] = res[i].replace(spotter_prompt, '')
        processed_sample['overall_caption'] = res[i]
        batch_processed_data.append(processed_sample)

    return batch_processed_data


def main(config_path, chunk_index, chunk_num, node_index, node_num):
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    lvlm = config['model_path']
    model_path = f"{config['ckpt_path']}/{lvlm}"
    print(f"loading {model_path}")
    if "llava" in lvlm and '1.6' in lvlm:
        model_name = get_model_name_from_path(model_path)
    elif "llava" in lvlm:
        model_name = lvlm
    else:
        raise ValueError(f"lvlm {lvlm} not supported")
    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    llava_model.eval().cuda()

    img_key = config['img_key']
    batch_size = config['batch_size']
    source_data_files, target_data_files = get_data_files(config, node_index=node_index, node_num=node_num)

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
            batch_processed_data = []

            with torch.no_grad():
                batch_processed_data = spotter_llava(llava_model, batch, image_processor, tokenizer, img_key)
                processed_data.extend(batch_processed_data)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        base_path = os.path.basename(source_data_file)
        output_path = f"detail_caption_construction/data/processed_data/{base_path.split('.')[0]}_chunk{chunk_index}.parquet"
        processed_df.to_parquet(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_index', type=int)
    parser.add_argument('--chunk_num', type=int, default=8)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=1)    
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        chunk_index=args.chunk_index, 
        chunk_num=args.chunk_num,
        node_index=args.node_index,
        node_num=args.node_num
    )





















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


import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import io
import torch
import pandas as pd
import numpy as np
import yaml
import tqdm
import argparse
from PIL import Image

import sys
sys.path.append('./reservoir/llava_code_base')
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates
from detail_caption_construction.utils.utils import get_data_files


def process_batch(model, batch, image_processor, tokenizer):
    def get_area(coordinates):
        return (coordinates[3] - coordinates[1]) * (coordinates[2] - coordinates[0])

    def get_cropped_images_and_indices(batch):
        all_image = []
        all_processed_boxes = []
        all_cropped_images = []
        all_cropped_images_indices = []
        for index, sample in batch.iterrows():
            image = np.array(Image.open(io.BytesIO(sample['frame'])).convert("RGB"))
            boxes = [[round(coordinate) for coordinate in coordinates] for coordinates in sample['SAM_cropped_boxes']]

            processed_boxes = []
            cropped_images = []
            for box in boxes:
                if True:    # get_area(box) < 7000:
                    dilate_x, dilate_y = sample['hw'][1] // 50, sample['hw'][0] // 50, 
                    box[0], box[1], box[2], box[3] = max(0, box[0]-dilate_x), max(0, box[1]-dilate_y), min(image.shape[1] - 1, box[2]+dilate_x), min(image.shape[0] - 1, box[3]+dilate_y)
                if box[2] - box[0] <= 2 or box[3] - box[1] <= 2:
                    continue
                processed_boxes.append(box)
                cropped_images.append(image[box[1]: box[3], box[0]: box[2], :])
            all_processed_boxes.append(processed_boxes)
            cropped_images = [Image.fromarray(img.astype('uint8')).convert('RGB') for img in cropped_images]  
            all_image.append(image)
            all_cropped_images.extend(cropped_images)
            all_cropped_images_indices.extend([index]*len(cropped_images))
        
        return all_image, all_processed_boxes, all_cropped_images, all_cropped_images_indices

    
    all_image, all_processed_boxes, all_cropped_images, all_cropped_images_indices = get_cropped_images_and_indices(batch)
    llava_ori_prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n'
    spotter_prompt = ' describe this picture in detail with no more than twenty words. ASSISTANT:'
    final_prompt = llava_ori_prompt + spotter_prompt
    this_batch_size = 32
    all_res = []
    for offset in range(0, len(all_cropped_images), this_batch_size):
        batch_cropped_images = all_cropped_images[offset: offset + this_batch_size]
        image_tensor = process_images(batch_cropped_images, image_processor, None).to(model.dtype)
        input_ids = tokenizer_image_token(final_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(model.device)
        all_input_ids = torch.tile(input_ids, [len(batch_cropped_images), 1])
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output_ids = model.generate(
                    all_input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=64,
                    use_cache=True
                )
        res = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_res.extend(res)
    
    regrouped_all_res = [[] for _ in range(len(batch))]
    for index, res in enumerate(all_res):
        regrouped_all_res[all_cropped_images_indices[index]].append(res)
    
    return all_image, all_processed_boxes, regrouped_all_res


def prepare_model_for_inference(model, dtype):
    model.cuda()
    model.eval()
    if dtype is not None:
        model.to(dtype)


def visualize(image, boxes, semantic_tags, out_file='reservoir/temp.jpg', linewidth=6):
    import matplotlib.pyplot as plt

    image_h, image_w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(image_w/100, image_h/100), dpi=100)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image)

    draw_label_setting = {'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}
    color = np.random.rand(len(semantic_tags), 3)
    n_terms = 10
    delta_y = math.floor(image_h/25)
    for i, (box, label) in enumerate(zip(boxes, semantic_tags)):
        box = [round(i, 2) for i in box]
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], edgecolor=color[i], facecolor=(0,0,0,0), lw=linewidth))
        label = label.strip()
        label_ = label.split()
        if len(label_) < n_terms:
            ax.text(box[0], box[1], label, fontsize=6, bbox=draw_label_setting, verticalalignment='top', color="gray")
        else:
            n_labels = (len(label_)-1)//n_terms+1
            for label_idx in range(n_labels):
                start, end = label_idx * n_terms, min((n_terms * (label_idx+1), len(label_)))
                this_label = ' '.join(label_[start:end])
                this_y = box[1] + delta_y * label_idx
                ax.text(box[0], this_y, this_label, fontsize=6, bbox=draw_label_setting, verticalalignment='top', color="gray")

    plt.savefig(out_file, format='jpeg')
    plt.close()



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
            all_image, all_boxes, all_tags = process_batch(llava_model, batch, image_processor, tokenizer)
            for index, (image, boxes, tags) in enumerate(zip(all_image, all_boxes, all_tags)):
                sample = batch.loc[index].to_dict()
                local_caption = repr([(boxes[i], tags[i]) for i in range(len(boxes))])
                sample['local_caption'] = local_caption
                processed_data.append(sample)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        base_path = os.path.basename(source_data_file)
        output_path = f"detail_caption_construction/data/processed_data/{base_path.split('.')[0]}_chunk{chunk_index}.parquet"
        processed_df.to_parquet(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_index', type=int, default=0)
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







            # image = Image.open(io.BytesIO(image)).convert("RGB")














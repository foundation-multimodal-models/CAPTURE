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
import pandas as pd
import collections
import torch
import time
import argparse
import yaml
from cruise.data_module.utils import parse_data_source


def watch_and_upload(config_path, node_index, node_num):
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    source_data_files = os.listdir(config['source_path'])
    source_data_files = [f"{config['source_path']}/{path}" for path in source_data_files]
    source_data_files.sort()
    start, end = node_index * (len(source_data_files) // node_num), (node_index + 1) * (len(source_data_files) // node_num)
    if len(source_data_files) - end < len(source_data_files) // node_num:
        end = len(source_data_files)
    source_data_files = source_data_files[start: end]

    target_data_files = os.listdir(f"{config['target_path']}/")
    target_data_files = [f"{config['target_path']}/{path}" for path in target_data_files]
    target_data_files.sort()
    target_data_files = [file.split('/')[-1].split('.')[0] for file in target_data_files]

    source_data_files = [source_data_file for source_data_file in source_data_files if f"{source_data_file.split('/')[-1].split('.')[0]}_processed" not in target_data_files]
    print(f"source_data_files: {source_data_files}")
    print(f"target_data_files: {target_data_files}")
    source_data_file_prefixes = [source_data_file.split('/')[-1].replace('.parquet', '').replace('.snappy', '') for source_data_file in source_data_files]
    source_data_file_prefixes = set(source_data_file_prefixes)
    remain_file_num = len(source_data_file_prefixes)
    print(f'remain_file_num: {remain_file_num}')

    while remain_file_num > 0:
        print(time.time())
        print(f"remain_file_num: {remain_file_num}")
        files = ["detail_caption_construction/data/processed_data/" + file for file in os.listdir("detail_caption_construction/data/processed_data")]
        file_prefix_mapping = collections.defaultdict(list)

        for file in files:
            prefix = file.split("/")[-1].split("_chunk")[0]
            if prefix in source_data_file_prefixes:
                file_prefix_mapping[prefix].append(file)

        for prefix, file_group in file_prefix_mapping.items():
            if len(file_group) == torch.cuda.device_count():
                print(f"start processing {prefix} results")
                file_group.sort()
                print("relating chunk files: ")
                for file in file_group:
                    print(file)
                all_df = [pd.read_parquet(file) for file in file_group]

                # discard useless columns
                df = pd.concat(all_df).reset_index()
                if 'sort_index' in df.keys():
                    df = df.drop('sort_index', axis=1)
                if 'index' in df.keys():
                    df = df.drop('index', axis=1)
                if 'level_0' in df.keys():
                    df = df.drop('level_0', axis=1)
            
                output_file = f"{config['target_path']}/{prefix.split('/')[-1]}_processed.parquet"
                df.to_parquet(output_file)

                os.system(f"rm detail_caption_construction/data/processed_data/{prefix}*")
                remain_file_num -= 1
                print(f"finished processing {prefix} results")
                print("========================================")
        
        if remain_file_num > 0:
            time.sleep(10)

    # for path in source_data_files:
    #     os.system(f"rm {path}")
    #     print(f"prev processed data {path} removed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=10)
    args = parser.parse_args()
    watch_and_upload(args.config, args.node_index, args.node_num)

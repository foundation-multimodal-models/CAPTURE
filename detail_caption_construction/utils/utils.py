
import os


def get_data_files(config, node_index, node_num):
    source_path = config['source_path']
    source_data_files = os.listdir(source_path)
    source_data_files = [f"{source_path}/{path}" for path in source_data_files]
    source_data_files.sort()
    start, end = node_index * (len(source_data_files) // node_num), (node_index + 1) * (len(source_data_files) // node_num)
    if len(source_data_files) - end < len(source_data_files) // node_num:
        end = len(source_data_files)
    source_data_files = source_data_files[start: end]

    # os.makedirs(f"{config['target_path']}/node_{node_index}/", exist_ok=True)
    target_data_files = os.listdir(f"{config['target_path']}/")
    target_data_files = [f"{config['target_path']}/{path}" for path in target_data_files]
    target_data_files.sort()
    target_data_files = [file.split('/')[-1].split('.')[0] for file in target_data_files]
    print(f"processed_files: {target_data_files}")

    return source_data_files, target_data_files



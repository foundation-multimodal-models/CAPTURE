
set -ex

node_index=$1
node_num=$2
chunk_num=$(nvidia-smi | grep MiB | wc -l)

bash prepare.sh


for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    CUDA_VISIBLE_DEVICES=$chunk_index nohup python3 detail_caption_construction/generate_stage1_overall_caption.py \
        --config_path detail_caption_construction/config_llava15_7b_detailcaps_4870/stage1_overall_caption.yaml \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > detail_caption_construction/scripts_output/stage1_overall_caption_$chunk_index.log 2>&1 &
done

python3 detail_caption_construction/merge_results.py --config detail_caption_construction/config_llava15_7b_detailcaps_4870/stage1_overall_caption.yaml --node_index $node_index --node_num $node_num > detail_caption_construction/scripts_output/watch_and_upload_stage1_overall_caption.log 2>&1
wait


for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    CUDA_VISIBLE_DEVICES=$chunk_index nohup python3 detail_caption_construction/generate_stage2_bbox.py \
        --config_path detail_caption_construction/config_llava15_7b_detailcaps_4870/stage2_bbox.yaml \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > detail_caption_construction/scripts_output/stage2_bbox_$chunk_index.log 2>&1 &
done

python3 detail_caption_construction/merge_results.py --config detail_caption_construction/config_llava15_7b_detailcaps_4870/stage2_bbox.yaml --node_index $node_index --node_num $node_num > detail_caption_construction/scripts_output/watch_and_upload_stage2_bbox.log 2>&1
wait


for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    CUDA_VISIBLE_DEVICES=$chunk_index nohup python3 detail_caption_construction/generate_stage3_local_caption.py \
        --config_path detail_caption_construction/config_llava15_7b_detailcaps_4870/stage3_local_caption.yaml \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > detail_caption_construction/scripts_output/stage3_local_caption_$chunk_index.log 2>&1 &
done

python3 detail_caption_construction/merge_results.py --config detail_caption_construction/config_llava15_7b_detailcaps_4870/stage3_local_caption.yaml --node_index $node_index --node_num $node_num > detail_caption_construction/scripts_output/watch_and_upload_stage3_local_caption.log 2>&1
wait


for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    CUDA_VISIBLE_DEVICES=$chunk_index nohup python3 detail_caption_construction/generate_stage4_filter.py \
        --config_path detail_caption_construction/config_llava15_7b_detailcaps_4870/stage4_filter.yaml \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > detail_caption_construction/scripts_output/stage4_filter_$chunk_index.log 2>&1 &
done

python3 detail_caption_construction/merge_results.py --config detail_caption_construction/config_llava15_7b_detailcaps_4870/stage4_filter.yaml --node_index $node_index --node_num $node_num > detail_caption_construction/scripts_output/watch_and_upload_stage4_filter.log 2>&1
wait


for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    CUDA_VISIBLE_DEVICES=$chunk_index nohup python3 detail_caption_construction/generate_stage5_caption_merge.py \
        --config_path detail_caption_construction/config_llava15_7b_detailcaps_4870/stage5_caption_merge.yaml \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > detail_caption_construction/scripts_output/stage5_caption_merge_$chunk_index.log 2>&1 &
done

python3 detail_caption_construction/merge_results.py --config detail_caption_construction/config_llava15_7b_detailcaps_4870/stage5_caption_merge.yaml --node_index $node_index --node_num $node_num > detail_caption_construction/scripts_output/watch_and_upload_stage5_caption_merge.log 2>&1
wait








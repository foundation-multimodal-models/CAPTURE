
if [ -d "./detail_caption_construction/LLaVA" ]; then
    echo "detail_caption_construction/LLaVA already configured, skipping......"
else
    cd detail_caption_construction
        git clone https://github.com/haotian-liu/LLaVA
        cd LLaVA
            pip3 install -e .
        cd ..
    cd ..
    location=$(pip3 show transformers | grep "Location")
    location=${location/Location: /}
    sudo chmod -R 777 ${location}/transformers/models/owlv2/
    rm ${location}/transformers/models/owlv2/image_processing_owlv2.py
    cp detail_caption_construction/utils/image_processing_owlv2.py ${location}/transformers/models/owlv2/
fi

sam_installed=$(pip3 list | grep segment)
if [ -n "$sam_installed" ]; then
    echo "sam already configured, skipping......"
else
    pip3 install git+https://github.com/facebookresearch/segment-anything.git
    pip3 install opencv-python pycocotools matplotlib onnxruntime onnx
fi

if [ -d "./detail_caption_construction/data" ]; then
    echo "detail_caption_construction/data already exists, skipping......"
else
    mkdir ./detail_caption_construction/data
fi

if [ -d "./detail_caption_construction/data/source_data" ]; then
    echo "detail_caption_construction data folders already configured, skipping......"
else
    cd detail_caption_construction
        cd data
            mkdir source_data
            mkdir stage1_overall_caption
            mkdir stage2_bbox
            mkdir stage3_local_caption
            mkdir stage4_filter
            mkdir stage5_caption_merge
            mkdir processed_data
        cd ..
    cd ..
fi

if [ -d "./detail_caption_construction/scripts_output" ]; then
    rm -r ./detail_caption_construction/scripts_output
    mkdir ./detail_caption_construction/scripts_output
else
    mkdir ./detail_caption_construction/scripts_output
fi


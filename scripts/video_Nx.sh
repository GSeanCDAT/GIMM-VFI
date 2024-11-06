SOURCE_PATH=$1
OUTPUT_PATH=$2
DS_FACTOR=$3
N=$4
#!/bin/bash
python ./src/video_Nx.py \
    --source-path $SOURCE_PATH \
    --output-path $OUTPUT_PATH \
    --ds-factor $DS_FACTOR \
    --N $N \
    -m='configs/gimmvfi/gimmvfi_r_arb.yaml' \
    -l='pretrained_ckpt/gimmvfi_r_arb_lpips.pt' \
    --eval

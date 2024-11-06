#!/bin/bash
python ./src/X4K.py \
    -m='configs/gimmvfi/gimmvfi_r_arb.yaml' \
    -l='pretrained_ckpt/gimmvfi_r_arb.pt' \
    --eval

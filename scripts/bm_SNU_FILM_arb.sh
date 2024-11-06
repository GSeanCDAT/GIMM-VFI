#!/bin/bash
python ./src/SNU_FILM_arb.py \
    -m='configs/gimmvfi/gimmvfi_r_arb.yaml' \
    -l='pretrained_ckpt/gimmvfi_r_arb.pt' \
    --eval

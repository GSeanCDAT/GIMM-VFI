MODEL_CONFIG=$1
OUTPUT=$2
CKPT=$3
NPROC_PER_NODE=$4

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
                                   --nnodes=1 \
                                   --master_port=16890 \
                                   --node_rank=0 \
                                   ./src/main.py \
                                   -m=$MODEL_CONFIG \
                                   -r=$OUTPUT \
                                   --nproc_per_node=$NPROC_PER_NODE \
                                   --nnodes=1 \
                                   --node_rank=0 \
                                   --seed=0 \
                                   -l=$CKPT 

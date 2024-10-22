# meta-llama/Llama-2-7b-hf
# meta-llama/Meta-Llama-3-8B

python train_opt.py \
        --model facebook/opt-2.7b \
        --save-dir dir/to/save/sliced_model/in \
        --sparsity 0.5 \
        --seed 1 \
        --device cuda:0 \
        --eval-baseline \
        --prune-method slice \
        --cal-batch-size 1 \
        --cal-nsamples 128 \
        --cal-max-seqlen 1024 \
        --lr 1e-2 \
        --epochs 10 \
        --num-Q 1 \
        --no-wandb \
        # --cal-dataset c4 \
        # --test-dataset c4
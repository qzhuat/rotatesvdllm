    # python run_lm_eval.py \
    #     --no-wandb

python run_baseline.py \
        --model facebook/opt-125m \
        --save-dir dir/to/save/sliced_model/in \
        --sparsity 0.50 \
        --device cuda:0 \
        --eval-baseline \
        --prune_method magnitude \
        --cal-batch-size 1 \
        --cal-nsamples 128 \
        --cal-max-seqlen 2048 \
        --cal-dataset c4 \
        --no-wandb
                
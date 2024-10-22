# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil

import torch
import torch.optim.lr_scheduler as lr_scheduler
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler

import numpy as np
from copy import deepcopy

import cayley_optimize.stiefel_optimizer as stiefel_optimizer

from lib.prune_opt import prune_wanda, prune_magnitude, prune_slice, prune_sparsegpt, prune_admm
from lib.eval import eval_ppl, eval_zero_shot, eval_ppl_wikitext_train, ppl_train, train_layer_by_layer
from lib.data import get_loaders 

def random_orthogonal(dim, A=None):
    """
    Create a orthogonal matrix
    """
    A = np.random.rand(dim, dim)
    Q, _ = np.linalg.qr(A)
    return torch.from_numpy(Q)


def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--cal-max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--final-orientation",
        type=str,
        default="random",
        choices=["random", "pca"],
        help="Final orientation of the sliced weights.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="rotpruner", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )
    
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=20, help="Epoch for training.")
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "slice"])

    return parser.parse_args() if interactive else parser.parse_args('')


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def slicing_main(args: argparse.Namespace) -> None:
    logging.info("Running Rotpruner experiment.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )

    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    # dataset = data_utils.get_dataset(args.cal_dataset)
    # train_dataset, test_dataset = dataset["train"], dataset["test"]
    # train_loader = data_utils.prepare_dataloader(
    #     dataset=train_dataset,
    #     tokenizer=tokenizer,
    #     max_seqlen=args.cal_max_seqlen,
    #     batch_size=args.cal_batch_size,
    #     nsamples=args.cal_nsamples,
    #     varied_seqlen=args.varied_seqlen,
    #     seed=args.seed,
    # )
    # test_loader = data_utils.prepare_test_dataloader(
    #     dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    # )

    dataset = data_utils.get_dataset(args.cal_dataset)
    train_dataset = dataset["train"]
    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    
    test_dataset = data_utils.get_dataset(args.test_dataset)
    test_dataset = test_dataset["test"]
    
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.sliced_model_path or args.ppl_only:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return
    
    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    model.to(config.device)
    # if args.prune_method == 'magnitude':
    #     prune_magnitude(args, model_adapter, train_loader, args.sparsity, device=config.device)
    # elif args.prune_method == 'wanda':
    #     prune_wanda(args=args, model_adapter=model_adapter, dataloader=train_loader, sparsity_ratio=args.sparsity, device=config.device)
    # elif args.prune_method == 'sparsegpt':
    #     prune_sparsegpt(args=args, model_adapter=model_adapter, dataloader=train_loader, sparsity_ratio=args.sparsity, device=config.device)
    
    prune_admm(args=args, model_adapter=model_adapter, dataloader=train_loader, sparsity_ratio=args.sparsity, device=config.device)
    
    model.to(config.device)
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'{args.prune_method} ppl: {dataset_ppl:.4f}')

if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)

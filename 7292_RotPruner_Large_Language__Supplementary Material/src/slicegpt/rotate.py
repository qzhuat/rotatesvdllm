# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter, srste_mask
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors

def find_layers(module, layers=[nn.Linear], name='', excludes=[]):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    for exclude in excludes:
        if exclude in name:
            return {}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1, excludes=excludes,
        ))
    return res

def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor, device="cpu", Q_dtype=torch.float64) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=Q_dtype)
        W.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor, device="cpu", Q_dtype=torch.float64) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=Q_dtype)
    W.weight.data = torch.matmul(Q.T, W_).to(device=device, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=Q_dtype)
        W.bias.data = torch.matmul(Q.T, b).to(device=device, dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor, device="cpu", Q_dtype=torch.float64) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=Q_dtype)
        W.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor, device="cpu", Q_dtype=torch.float64) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=Q_dtype)
    W.weight.data = torch.matmul(Q.T, W_).to(device=device, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=Q_dtype)
        W.bias.data = torch.matmul(Q.T, b).to(device=device, dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor, device="cpu", Q_dtype=torch.float64) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=Q_dtype)
        W.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]


def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor, device="cpu", Q_dtype=torch.float64) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=Q_dtype)
    W.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension


def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    # dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    Qs, 
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    Q_dtype = Qs[0].dtype

    # inps, args, kwargs, ignore_masks = [], [], [], []
    # for batch in dataloader:
    #     inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
    #     inps.append(inp_batch)
    #     args.append(args_batch)
    #     kwargs.append(kwargs_batch)
    #     if apply_mask:
    #         ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # rotate and slice embeddings
    # eig_val, Q = pca_calc(inps, ignore_masks)
    # Q = Q.to(device=config.device)
    # if final_orientation == 'random':
    #     R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
    #     Q = Q @ R.to(Q.device)
    i=0
    Q = Qs[i]
    rotate_embeddings(model_adapter, Q, Q_dtype=Q_dtype)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        # import pdb; pdb.set_trace()
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q, Q_dtype=Q_dtype)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        # for i, inp in enumerate(inps):
        #     args[i] = layer_adapter.get_updated_args(
        #         torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
        #             :, :, : slicing_scheduler.get_attention_input_dimension(idx)
        #         ].cpu(),
        #         args[i],
        #     )

        # mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        # eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        # Q = Q.to(device=config.device, dtype=torch.float64)
        # if final_orientation == 'random':
        #     R = random_orthogonal_upper_left(
        #         Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        #     )
        #     Q = Q @ R.to(Q.device)
        i += 1
        Q = Qs[i]

        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q, Q_dtype=Q_dtype)
        slice_attention_output(
            layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q, Q_dtype=Q_dtype)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        # _, inps = get_signals(layer_adapter, args, kwargs)
        # eig_val, Q = pca_calc(inps, ignore_masks)
        # if final_orientation == 'random':
        #     R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
        #     Q = Q @ R.to(Q.device)
        i += 1
        Q = Qs[i]
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q, Q_dtype=Q_dtype)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q, Q_dtype=Q_dtype)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This version works for models where the MLP block and the attention block are computed in parallel.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # rotate and slice embeddings
    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # update the input signals to this layer, and re-run it
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        # the simpler equivalent of get_signals
        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = pca_calc(inps, ignore_masks)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        # update shortcut matrix
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        # slice the shortcut (there is only one, we use attn_shortcut buffer)
        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")

@torch.no_grad
def get_rotate_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    
    Return the rotation matrix.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch, _ in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)
    Qs = []
    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    Qs.append(Q.to(dtype))
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(
                Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            )
            Q = Q @ R.to(Q.device)
        Qs.append(Q.to(dtype))
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(
            layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(inps, ignore_masks)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)
        Qs.append(Q.to(dtype))
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    # if slicing_scheduler.do_slice_head:
        # slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")
    return Qs

@torch.no_grad
def rotate_sequential(
    model_adapter: ModelAdapter,
    Qs: list,
    dif_Q: bool = True,
    num: int = 1,
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """    
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    Q_type = Qs[0].dtype
    device = model_adapter.model.device

    layers = model_adapter.get_layers()
    
    # assert len(Qs) == 2 * len(layers) + 1

    # rotate and slice embeddings
    i = 0
    Q = Qs[i]
    rotate_embeddings(model_adapter, Q, device=device, Q_dtype=Q_type)

    logging.info("Rotate layers")
    
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layer.attn_shortcut_Q.weight.data = Q.to(dtype=dtype)
        # layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q, device=device, Q_dtype=Q_type)
        i += 1
        Q = Qs[i//num]

        layer.attn_shortcut_Q.weight.data = torch.matmul(Q.T.to(dtype=dtype), layer.attn_shortcut_Q.weight.data)
        # layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))
        
        rotate_attention_output(layer_adapter, Q, device=device, Q_dtype=Q_type)

        layer.mlp_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layer.mlp_shortcut_Q.weight.data = Q.to(dtype=dtype)
        # layer.mlp_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        rotate_mlp_input(layer_adapter, Q, device=device, Q_dtype=Q_type)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        i += 1
        Q = Qs[i//num]

        layer.mlp_shortcut_Q.weight.data = torch.matmul(Q.T.to(dtype=dtype), layer.mlp_shortcut_Q.weight.data)
        # layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))
        
        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q, device=device, Q_dtype=Q_type)

        # Run GC and cleanup GPU memory
        cleanup_memory()
        

    # rotate and slice head
    rotate_head(model_adapter, Q, device=device, Q_dtype=Q_type)

    # update model's slicing config
    logging.info("Rotate layers done")

@torch.no_grad  
def rotate_and_mask_sequential(
    model_adapter: ModelAdapter,
    Qs: list,
    W_masks: list,
    dif_Q: bool = True,
    num: int = 1,
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """    
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    Q_type = Qs[0].dtype
    device = model_adapter.model.device

    layers = model_adapter.get_layers()
    
    # assert len(Qs) == 2 * len(layers) + 1
    assert len(W_masks) == len(layers)

    # rotate and slice embeddings
    Q = Qs[0]
    rotate_embeddings(model_adapter, Q, device=device, Q_dtype=Q_type)

    logging.info("Rotate layers")
    i = 1
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layer.attn_shortcut_Q.weight.data = Q.to(dtype=dtype)

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q, device=device, Q_dtype=Q_type)
        for name, module in layer_adapter.get_attention_inputs_dict().items():
            module.weight.data[W_masks[idx][name]] = 0

        Q = Qs[i//num]

        layer.attn_shortcut_Q.weight.data = torch.matmul(Q.T.to(dtype=dtype), layer.attn_shortcut_Q.weight.data)
        layer.attn_shortcut_Q.weight.data[W_masks[idx]["attn_shortcut_Q"]] = 0

        rotate_attention_output(layer_adapter, Q, device=device, Q_dtype=Q_type)
        for name, module in layer_adapter.get_attention_outputs_dict().items():
            module.weight.data[W_masks[idx][name]] = 0

        layer.mlp_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layer.mlp_shortcut_Q.weight.data = Q.to(dtype=dtype)

        rotate_mlp_input(layer_adapter, Q, device=device, Q_dtype=Q_type)
        for name, module in layer_adapter.get_mlp_inputs_dict().items():
            module.weight.data[W_masks[idx][name]] = 0

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        if dif_Q:
            i += 1
        Q = Qs[i//num]

        layer.mlp_shortcut_Q.weight.data = torch.matmul(Q.T.to(dtype=dtype), layer.mlp_shortcut_Q.weight.data)
        layer.mlp_shortcut_Q.weight.data[W_masks[idx]["mlp_shortcut_Q"]] = 0

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q, device=device, Q_dtype=Q_type)
        for name, module in layer_adapter.get_mlp_outputs_dict().items():
            module.weight.data[W_masks[idx][name]] = 0

        # Run GC and cleanup GPU memory
        cleanup_memory()
        i += 1

    # rotate and slice head
    rotate_head(model_adapter, Q, device=device, Q_dtype=Q_type)

    # update model's slicing config
    logging.info("Rotate layers done")
    
# def get_next_layer(
#     model, 
#     i: int,
# ):
#     layers = model.model.decoder.layers
#     if i == 0: # rotate emb and attn_in
#         return 1
#     elif i >= 2 * len(layers) - 2: # rotate mlp_out and head
#         return -1
#     else: # rotate mlp_out and attn_in
#         return i//2 + 1

@torch.no_grad    
def get_rotate_matrix(
    model_adapter: ModelAdapter,
    i: int,
) -> dict:
    model_adapter.model.eval()

    layers = model_adapter.get_layers()
    subset = {}
    if i == 0: # rotate emb and attn_in
        subset.update(layers[i].get_attention_inputs_dict())
        
    elif i == 2 * len(layers): # rotate mlp_out and head
        subset.update(layers[i//2-1].get_mlp_outputs_dict())
        subset.update({'mlp_shortcut_Q': layers[i//2-1].layer.mlp_shortcut_Q})

    elif i % 2: # rotate attn_out and mlp_in
        subset.update(layers[i//2].get_attention_outputs_dict())
        subset.update(layers[i//2].get_mlp_inputs_dict())
        subset.update({'attn_shortcut_Q': layers[i//2].layer.attn_shortcut_Q})

    else: # rotate mlp_out and attn_in
        subset.update(layers[i//2-1].get_mlp_outputs_dict())
        subset.update(layers[i//2].get_attention_inputs_dict())
        subset.update({'mlp_shortcut_Q': layers[i//2-1].layer.mlp_shortcut_Q})
    
    return subset

@torch.no_grad    
def rotate_one(
    model_adapter: ModelAdapter,
    Qs: list,
    i: int,
) -> dict:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    device = model_adapter.model.device
    dtype = next(iter(model_adapter.model.parameters())).dtype
    Q_type = Qs[0].dtype

    layers = model_adapter.get_layers()
    subset = {}
    
    Q = Qs[i]
    if i == 0: # rotate emb and attn_in
        rotate_embeddings(model_adapter, Q, device=device, Q_dtype=Q_type)
        rotate_attention_inputs(layers[i], Q, device=device, Q_dtype=Q_type)
        
        layers[i].layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layers[i].layer.attn_shortcut_Q.weight.data = Q.to(dtype=dtype)
        
        # layers[i].layer.attn_shortcut_Q = Q.T.to(dtype=dtype)
        
        subset.update(layers[i].get_attention_inputs_dict())
        # subset.update({'attn_shortcut_Q': layers[i].attn_shortcut_Q})
        
    elif i == 2 * len(layers): # rotate mlp_out and head
        rotate_mlp_output(layers[i//2-1], Q, device=device, Q_dtype=Q_type)
        rotate_head(model_adapter, Q, device=device, Q_dtype=Q_type)
        
        layers[i//2-1].layer.mlp_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layers[i//2-1].layer.mlp_shortcut_Q.weight.data = torch.matmul(Qs[i-1].T, Q.to(dtype=dtype)).T
        
        subset.update(layers[i//2-1].get_mlp_outputs_dict())
        subset.update({'mlp_shortcut_Q': layers[i//2-1].layer.mlp_shortcut_Q})

    elif i % 2: # rotate attn_out and mlp_in
        rotate_attention_output(layers[i//2], Q, device=device, Q_dtype=Q_type)
        rotate_mlp_input(layers[i//2], Q, device=device, Q_dtype=Q_type)

        layers[i//2].layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layers[i//2].layer.attn_shortcut_Q.weight.data = torch.matmul(Qs[i-1].T, Q.to(dtype=dtype)).T
        layers[i//2].layer.mlp_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layers[i//2].layer.mlp_shortcut_Q.weight.data = Q.to(dtype=dtype)
        
        subset.update(layers[i//2].get_attention_outputs_dict())
        subset.update(layers[i//2].get_mlp_inputs_dict())
        subset.update({'attn_shortcut_Q': layers[i//2].layer.attn_shortcut_Q})
        # subset.update({'mlp_shortcut_Q': layers[i//2].mlp_shortcut_Q})

    else: # rotate mlp_out and attn_in
        rotate_mlp_output(layers[i//2-1], Q, device=device, Q_dtype=Q_type)
        rotate_attention_inputs(layers[i//2], Q, device=device, Q_dtype=Q_type)
        
        layers[i//2-1].layer.mlp_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layers[i//2-1].layer.mlp_shortcut_Q.weight.data = torch.matmul(Qs[i-1].T, Q.to(dtype=dtype)).T
        layers[i//2].layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        layers[i//2].layer.attn_shortcut_Q.weight.data = Q.to(dtype=dtype)
        
        subset.update(layers[i//2-1].get_mlp_outputs_dict())
        subset.update(layers[i//2].get_attention_inputs_dict())
        subset.update({'mlp_shortcut_Q': layers[i//2-1].layer.mlp_shortcut_Q})
        # subset.update({'attn_shortcut_Q': layers[i//2].attn_shortcut_Q})
    
    return subset
    
@torch.no_grad    
def rotate_and_mask_one(
    model_adapter: ModelAdapter,
    Qs: list,
    i: int,
    W_masks: dict,
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    device = model_adapter.model.device
    dtype = next(iter(model_adapter.model.parameters())).dtype
    Q_type = Qs[0].dtype

    layers = model_adapter.get_layers()
    subset = {}
    
    Q = Qs[i]
    if i == 0: # rotate emb and attn_in
        rotate_embeddings(model_adapter, Q, device=device, Q_dtype=Q_type)
        
        rotate_attention_inputs(layers[i], Q, device=device, Q_dtype=Q_type)  
        for name, module in layers[i].get_attention_inputs_dict().items():
            module.weight.data[W_masks[name]] = 0
            
        # layers[i].layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        # layers[i].layer.attn_shortcut_Q.weight.data = Q.to(dtype=dtype)
        layers[i].layer.attn_shortcut_Q = Q.T.to(dtype=dtype)
            
    elif i == 2 * len(layers): # rotate mlp_out and head
        rotate_mlp_output(layers[i//2-1], Q, device=device, Q_dtype=Q_type)
        for name, module in layers[i//2-1].get_mlp_outputs_dict().items():
            module.weight.data[W_masks[name]] = 0
            
        rotate_head(model_adapter, Q, device=device, Q_dtype=Q_type)
        
        # layers[i//2-1].layer.mlp_shortcut_Q.weight.data = torch.matmul(layers[i//2-1].layer.mlp_shortcut_Q.weight.data, Q.to(dtype=dtype)).T
        layers[i//2-1].layer.mlp_shortcut_Q = torch.matmul(Qs[i-1].T, Q)
        
        layers[i//2-1].layer.mlp_shortcut_Q[W_masks['mlp_shortcut_Q'].T] = 0
        
    elif i % 2: # rotate attn_out and mlp_in
        rotate_attention_output(layers[i//2], Q, device=device, Q_dtype=Q_type)
        for name, module in layers[i//2].get_attention_outputs_dict().items():
            module.weight.data[W_masks[name]] = 0
        
        rotate_mlp_input(layers[i//2], Q, device=device, Q_dtype=Q_type)
        for name, module in layers[i//2].get_mlp_inputs_dict().items():
            module.weight.data[W_masks[name]] = 0
            
        # layers[i//2].layer.attn_shortcut_Q.weight.data = torch.matmul(layers[i//2].layer.attn_shortcut_Q.weight.data, Q.to(dtype=dtype)).T
        layers[i//2].layer.attn_shortcut_Q = torch.matmul(Qs[i-1].T, Q)
        
        # layers[i//2].layer.mlp_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        # layers[i//2].layer.mlp_shortcut_Q.weight.data = Q.to(dtype=dtype)
        layers[i//2].layer.mlp_shortcut_Q = Q.T
        
        layers[i//2].layer.attn_shortcut_Q[W_masks['attn_shortcut_Q'].T] = 0
        
    else: # rotate mlp_out and attn_in
        rotate_mlp_output(layers[i//2-1], Q, device=device, Q_dtype=Q_type)
        for name, module in layers[i//2-1].get_mlp_outputs_dict().items():
            module.weight.data[W_masks[name]] = 0
        
        rotate_attention_inputs(layers[i//2], Q, device=device, Q_dtype=Q_type)
        for name, module in layers[i//2].get_attention_inputs_dict().items():
            module.weight.data[W_masks[name]] = 0
        
        # layers[i//2-1].layer.mlp_shortcut_Q.weight.data = torch.matmul(layers[i//2-1].layer.mlp_shortcut_Q.weight.data, Q.to(dtype=dtype)).T
        layers[i//2-1].layer.mlp_shortcut_Q = torch.matmul(Qs[i-1].T, Q)
        
        # layers[i//2].layer.attn_shortcut_Q = nn.Linear(Q.shape[0], Q.shape[1], bias=False)
        # layers[i//2].layer.attn_shortcut_Q.weight.data = Q.to(dtype=dtype)
        layers[i//2].layer.attn_shortcut_Q = Q.T
        layers[i//2-1].layer.mlp_shortcut_Q[W_masks['mlp_shortcut_Q'].T] = 0
    
def rotate_and_mask_one_implicit(
    model_adapter: ModelAdapter,
    Qs: list,
    i: int,
    W_masks: list = None,
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    device = model_adapter.model.device
    dtype = next(iter(model_adapter.model.parameters())).dtype

    layers = model_adapter.get_layers()
    
    Q = Qs[i]
    if i == 0: # rotate emb and attn_in
        layers[0].layer.embed_Q = Q
        for name, module in layers[i//2].get_attention_inputs_dict().items():
            module.ori = 1
            module.Q = Q
            module.mask = W_masks[name]
        layers[i].layer.attn_shortcut_Q = Q.T
        
    elif i == 2 * len(layers): # rotate mlp_out and head
        for name, module in layers[i//2-1].get_mlp_outputs_dict().items():
            module.ori = 0
            module.Q = Q
            module.mask = W_masks[name]
        layers[-1].layer.head_Q = Q.T
        layers[i//2-1].layer.mlp_shortcut_Q = torch.matmul(Qs[i-1].T, Q)
        layers[i//2-1].layer.mlp_shortcut_Q = srste_mask(layers[i//2-1].layer.mlp_shortcut_Q, W_masks['mlp_shortcut_Q'].T)
    elif i % 2: # rotate attn_out and mlp_in
        for name, module in layers[i//2].get_attention_outputs_dict().items():
            module.ori = 0
            module.Q = Q
            module.mask = W_masks[name]
        for name, module in layers[i//2].get_mlp_inputs_dict().items():
            module.ori = 1
            module.Q = Q
            module.mask = W_masks[name]
        layers[i//2].layer.attn_shortcut_Q = torch.matmul(Qs[i-1].T, Q)
        layers[i//2].layer.attn_shortcut_Q = srste_mask(layers[i//2].layer.attn_shortcut_Q, W_masks['attn_shortcut_Q'].T)
        
        layers[i//2].layer.mlp_shortcut_Q = Q.T
    else: # rotate mlp_out and attn_in
        for name, module in layers[i//2-1].get_mlp_outputs_dict().items():
            module.ori = 0
            module.Q = Q
            module.mask = W_masks[name]
        for name, module in layers[i//2].get_attention_inputs_dict().items():
            module.ori = 1
            module.Q = Q
            module.mask = W_masks[name]
        layers[i//2-1].layer.mlp_shortcut_Q = torch.matmul(Qs[i-1].T, Q)
        layers[i//2-1].layer.mlp_shortcut_Q = srste_mask(layers[i//2-1].layer.mlp_shortcut_Q, W_masks['mlp_shortcut_Q'].T)
        layers[i//2].layer.attn_shortcut_Q = Q.T
    
def rotate_and_mask_implicit_sequential(
    model_adapter: ModelAdapter,
    Qs: list,
    W_masks: list = None,
    mask_shortcut: bool = True,
    dif_Q: bool = True,
    num: int = 1,
) -> None:
    """
    Rotate implicitly the provided model.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    device = model_adapter.model.device

    layers = model_adapter.get_layers()
    
    # assert len(Qs) == 2 * len(layers) + 1
    assert len(W_masks) == len(layers)

    # rotate and slice embeddings
    Q = Qs[0]
    
    layers[0].layer.embed_Q = Q
    # rotate_embeddings(model_adapter, Q, device=device)
    i = 1
    for idx, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = Q.T

        # rotate and slice the attention inputs to match previous layer
        for name, module in layer_adapter.get_attention_inputs_dict().items():
            module.ori = 1
            module.Q = Q
            module.mask = W_masks[idx][name]

        Q = Qs[i//num]
        layer.attn_shortcut_Q = torch.matmul(layer.attn_shortcut_Q, Q)
        if mask_shortcut:
            layer.attn_shortcut_Q = srste_mask(layer.attn_shortcut_Q.T, W_masks[idx]['attn_shortcut_Q']).T
        for name, module in layer_adapter.get_attention_outputs_dict().items():
            module.ori = 0
            module.Q = Q
            module.mask = W_masks[idx][name]

        layer.mlp_shortcut_Q = Q.T

        for name, module in layer_adapter.get_mlp_inputs_dict().items():
            module.ori = 1
            module.Q = Q
            module.mask = W_masks[idx][name]

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        if dif_Q:
            i += 1
        Q = Qs[i//num]

        layer.mlp_shortcut_Q = torch.matmul(layer.mlp_shortcut_Q, Q)
        if mask_shortcut:
            layer.mlp_shortcut_Q = srste_mask(layer.mlp_shortcut_Q.T, W_masks[idx]['mlp_shortcut_Q']).T

        # optionally slice the mlp/head connection in the last layer
        for name, module in layer_adapter.get_mlp_outputs_dict().items():
            module.ori = 0
            module.Q = Q
            module.mask = W_masks[idx][name]

        # Run GC and cleanup GPU memory
        cleanup_memory()
        i += 1

    # rotate and slice head
    layers[-1].layer.head_Q = Q.T

@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model_adapter.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model_adapter, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer_adapter, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        # slice attn weights 2nd dim, attn shortcut 1st dim
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))

        # slice mlp input 2nd dimension
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))

        # slice mlp shortcut 1st dimension
        # slice mlp shortcut
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])

        # slice mlp weights 1st dimension
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:  # parallel case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:  # sequential case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )

            # slice attention weights 1st dimension
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec

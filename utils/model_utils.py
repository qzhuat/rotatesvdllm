#coding:utf8
import os
import sys
import torch
import torch.nn as nn
from slicegpt.adapters.llama_adapter import rot_mask_Linear

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear, rot_mask_Linear], name=''):
    # type(module)
    # print(type(module))
    # print(name)
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
# def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
#     type(module)
#     print(type(module))
#     if type(module) in layers:
#         return {name: module}
#     res = {}
#     for name1, child in module.named_children():
#         res.update(find_layers(
#             child, layers=layers, name=name + '.' + name1 if name != '' else name1
#         ))
#     return res
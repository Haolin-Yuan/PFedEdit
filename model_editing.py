import copy

import torch
import torch.nn as nn
from collections import OrderedDict
from model_util import compare_models

model_hook = {}

# def get_flatten_features(data):
#     # not using data shape since batch size = 1
#     data = data.view(1,-1)
#     return data

def forward_hook(module_name):
    def hook(module, input, output):
        model_hook[module_name] = output
        for i in model_hook[module_name]:
            if i != None: i.detach().cpu()
    return hook

def register(model):
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook(name))

def hook_to_cpu():
    for key in model_hook.keys():
        if isinstance(model_hook[key], tuple):
            model_hook[key] = (model_hook[key][0].detach().cpu(), model_hook[key][1])
        elif model_hook[key] != None:
            model_hook[key] = model_hook[key].detach().cpu()

def get_ori_weights_copy(model, layer_name):
    '''
    :return: copy of original layer weights
    '''
    print("private local layer:  ", layer_name)
    for layer in model.state_dict():
        if layer_name in layer:
            # weights_copy = torch.zeros(model.state_dict()[f"{layer_name}.weight"].shape)
            # return weights_copy.copy_(model.state_dict()[f"{layer_name}.weight"])
            return model.state_dict()[f"{layer_name}.weight"].detach().clone()


def get_kv_stats(layer_name, model, data_loader, device):
    '''
    get v stats using clean light skin samples
    '''
    hooks = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            _ = model(x)
            hook_to_cpu()
            hooks.append(model_hook.copy())
    result = []
    for i in range(len(data_loader)):
        result.append(hooks[i][layer_name])
    model.to("cpu")
    return result

def get_kv_layer_name(model, layer_name):
    layer = []
    for key, _ in model.state_dict().items():
        layer.append((".").join(key.split(".")[:-1]))
    key_list = list(OrderedDict.fromkeys(layer))
    for i, value in enumerate(key_list):
        if layer_name == value:
            mlp_k_layer = key_list[i-1]
    return mlp_k_layer, layer_name

def get_cov_inv(layer_name, model, data_loader, device):

    model.to(device)
    model.eval()
    hooks = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            _ = model(x)
            hook_to_cpu()
            hooks.append(model_hook.copy())

    mlp_prev_layer = layer_name
    # c_shape = hooks[0][mlp_prev_layer].squeeze().shape[0]
    c_shape = hooks[0][mlp_prev_layer].squeeze().shape[1]
    mom = torch.zeros(c_shape, c_shape) # (50,50)
    for i in range(len(data_loader)):
        a = hooks[i][mlp_prev_layer].squeeze()
        # mom += a.mm(a.t())    # (197, 197) 1.0,2.0
        mom += a.t().mm(a)      #3072, 3072 3.0
    mom = torch.inverse(mom/len(data_loader))
    model.to("cpu")

    return mom/mom.norm()


def insert_new_weights(model, layer_name, new_weights, device):
    mlp_layer = int(layer_name.split(".")[-3].split("_")[-1])
    mlp_index = int(layer_name.split(".")[-1])
    # a n*m linear layer should be declared as Linear(m, n)
    with torch.no_grad():
        new_linear = nn.Linear(new_weights.shape[1], new_weights.shape[0])
        new_linear.weight.copy_(new_weights)
        model.model.encoder.layers[mlp_layer].mlp[mlp_index] = new_linear.to(device)
    return model.to(device)

def apply_model_editing(model, prev_model, layer_name, data, device):
        print("*********************** Start Model Editing ***********************")
        # for each MLP block, edit W_proj
        ori_weights = get_ori_weights_copy(model, layer_name).to(device)

        # W shape (768, 3072)
        register(model)
        register(prev_model)
        k_name, v_name = get_kv_layer_name(model, layer_name)
        print(k_name, v_name)
        cov_inv = get_cov_inv(layer_name=k_name, model=prev_model, data_loader=data, device=device).to(device)
        K = get_kv_stats(k_name, model, data, device)
        V_target = get_kv_stats(v_name, model, data, device)
        V = get_kv_stats(v_name, prev_model, data, device)


        new_weights = ori_weights.detach().clone()
        with torch.no_grad():
            for i, d in enumerate(range(len(data))):
                k, v = K[i].squeeze().to(device), V[i].squeeze().to(device)
                v_target = V_target[i].squeeze().to(device)

                # 1.0
                # left_vector = cov_inv @ k    #(3072, 197)
                # right_vector = (v - k @ ori_weights.t()) / torch.dot(left_vector.view(1, -1).squeeze(),
                #                                                      k.view(1, -1).squeeze())
                # update = right_vector.t() @ left_vector


                # print(f"k {k.shape}, v {v.shape}, cov {cov_inv.shape}")
                # 2.0
                # left_vector = (cov_inv @ k).t() # 3072, 197
                # # (v - k @ ori_weights.t()) / (k @ left_vector)
                # right_vector = torch.inverse(k @ left_vector) @ (v - v_target) # (197, 197) @ (197,768) = (197,768)
                # update = left_vector @ right_vector # 3072, 768

                # 3.0
                update = (cov_inv @ k.t() @ v_target).t()

                new_weights += update

            # print(model.state_dict()[f"{layer_name}.weight"].shape)
        # new_weights = new_weights.t()     #2.0
        new_model = insert_new_weights(copy.deepcopy(model), layer_name, new_weights, device=device)
        compare_models(new_model.to(device), model.to(device))
        return new_model


def replace_model_layer(model_1, model_2, layer_name, device):
    new_weights = model_1.state_dict()[f"{layer_name}.weight"].detach().clone()
    new_model = insert_new_weights(copy.deepcopy(model_2), layer_name, new_weights, device=device)
    compare_models(model_2.to(device), new_model.to(device))
    return new_model


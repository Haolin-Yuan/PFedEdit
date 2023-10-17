import torch
import torch.nn as nn
from collections import OrderedDict

model_hook = {}

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
            # bias = model.state_dict()[f"{layer_name}.bias"].detach().clone()
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
            mlp_v_layer = key_list[i]
    return mlp_k_layer, mlp_v_layer

def get_cov_inv(layer_name, model, data_loader, device):
    '''
    gets k stats from dark skin sample data
    '''
    # stores all model hooks
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
    # c_shape = hooks[0][mlp_prev_layer].view(1,-1).shape[1]
    c_shape = hooks[0][mlp_prev_layer].squeeze().shape[0]
    mom = torch.zeros(c_shape, c_shape) # (50,50)
    for i in range(len(data_loader)):
        a = hooks[i][mlp_prev_layer].squeeze() #(50,768)
        mom += a.mm(a.t())
    mom = torch.inverse(mom/len(data_loader))
    model.to("cpu")
    return mom/mom.norm()


def insert_new_weights(model, layer_name, new_weights, device):
    mlp_layer = int(layer_name.split(".")[-3].split("_")[-1])
    mlp_index = int(layer_name.split(".")[-1])
    new_linear = nn.Linear(new_weights.shape[1], new_weights.shape[0])
    new_linear.weight.copy_(new_weights)
    model.model.encoder.layers[mlp_layer].mlp[mlp_index] = new_linear.to(device)
    return model

def apply_model_editing(model, prev_model, layer_name, data, device):
        print("*********************** Start Model Editing ***********************")
        ori_weights = get_ori_weights_copy(model, layer_name).to(device)

        # in feature 768, out 3072 (3072,768), bias (3072)
        register(model)
        register(prev_model)
        k_name, v_name = get_kv_layer_name(model, layer_name)
        cov_inv = get_cov_inv(layer_name=k_name, model=prev_model, data_loader=data, device=device).to(device)
        K = get_kv_stats(k_name, model, data, device)
        V = get_kv_stats(v_name, prev_model, data, device)

        new_weights = ori_weights.clone()
        with torch.no_grad():
            for i, d in enumerate(range(len(data))):
                k, v = K[i].squeeze().to(device), V[i].squeeze().to(device)  # (50, 768) (50, 3072)
                left_vector = cov_inv @ k  # cov_inv (50,50), left vector (50, 768)
                right_vector = (v - k @ ori_weights.t()) / torch.dot(left_vector.view(1, -1).squeeze(),
                                                                     k.view(1, -1).squeeze())
                # (50, 3072)
                update = right_vector.t() @ left_vector
                new_weights += update
            # print(model.state_dict()[f"{layer_name}.weight"].shape)
            # model.state_dict()[f"{layer_name}.weight"].data = nn.Parameter(new_weights, requires_grad = True).to(device)
            model = insert_new_weights(model, layer_name, new_weights, device=device)
        return model



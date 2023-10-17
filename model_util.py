import torch.nn as nn
from torch.nn.functional import softmax
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights
from collections import OrderedDict


# MODEL_PATH = "model_weights/ViT_model_B32.pt"
MODEL_PATH = "model_weights/ViT_model_B32_editing_first.pt"


def evaluate_global(users, test_dataloders, users_index):
    testing_corrects = 0
    testing_sum = 0
    for index in users_index:
        corrects, num = users[index].evaluate(test_dataloders[index])
        testing_corrects += corrects
        testing_sum += num
    print(f"Acc: {testing_corrects / testing_sum}")
    return (testing_corrects / testing_sum)


def compute_st_bias(prob, gt_label):
    gt_label = gt_label.detach().cpu()
    prob = prob.detach().cpu()
    bias = 0
    for i in range(prob.shape[0]):
        bias += prob[i][gt_label[i]]

    return bias / prob.shape[0]



class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        # self.weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights= ViT_B_16_Weights.IMAGENET1K_V1)
        self.heads_layers = OrderedDict()
        self.heads_layers['head'] = nn.Linear(768, num_classes)
        self.model.heads = nn.Sequential(self.heads_layers)

    def forward(self, x):
        logits = self.model(x)
        return logits, softmax(logits, dim=-1)


class ModelForwardHook():
    def __init__(self, device):
        self.model_hook = {}
        self.clean_hooks = []
        self.device = device

    def register_clean(self, module_name):
        def hook(module, input, output):
            self.model_hook[module_name] = output
            # for i in self.model_hook[module_name]:
            #     if i != None: i.detach().cpu()
        return hook

    def recover(self, module_name, index):
        def modified_hook(module, input, output):
            self.model_hook[module_name] = self.clean_hooks[index][module_name]
            if isinstance(self.model_hook[module_name], tuple):
                return (self.model_hook[module_name][0].to(self.device), self.model_hook[module_name][1])
            else:
                return self.model_hook[module_name].to(self.device)
        return modified_hook

    def collect_clean_hook(self):
        self.clean_hooks.append(self.model_hook)

    # def collect_corrupted_hook(self):
    #     self.corrupted_hooks.append(self.model_hook)

    def close(self):
        self.model_hook = {}
        self.clean_hooks = []
        self.corrupted_hooks = []


if __name__=="__main__":
    model = ViT(num_classes=4)
    print(model)
    print("***"*30)
    for name, module in model.named_modules():
        print(name)
    print("***"*30)
    for i in model.state_dict():
        print(i)



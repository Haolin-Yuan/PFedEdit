import torch.nn as nn
from torch.nn.functional import softmax
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights, resnet50, ResNet50_Weights
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import copy


# MODEL_PATH = "model_weights/ViT_model_B32.pt"
MODEL_PATH = "model_weights/ViT_model_B32_editing_first.pt"

torch.manual_seed(2023)


def agg_weights(weights):
    with torch.no_grad():
        weights_avg = copy.deepcopy(weights[0])
        for k in weights_avg.keys():
            for i in range(1, len(weights)):
                weights_avg[k] += weights[i][k]
            weights_avg[k] = torch.div(weights_avg[k], len(weights))
    return weights_avg

def custom_agg_weights(weights, layer_name):
    with torch.no_grad():
        weights_avg = copy.deepcopy(weights[0])
        for k in weights_avg.keys():
            count = 0
            for i in range(1, len(weights)):
                for j in range(len(layer_name[i])):
                    if layer_name[i][j] not in k:
                        weights_avg[k] += weights[i][k]
                        count += 1
            weights_avg[k] = torch.div(weights_avg[k], count)
    return weights_avg


def merge_two_model_weights(weights_1, weights_2, device):
    with torch.no_grad():
        weights_avg = copy.deepcopy(weights_1)
        for k in weights_avg.keys():
            weights_avg[k] += weights_2[k]
            weights_avg[k] = torch.div(weights_avg[k], 2)
    return weights_avg


def evaluate_global(users, test_dataloders, users_index):
    testing_corrects = 0
    testing_sum = 0
    for index in users_index:
        corrects, num = users[index].evaluate(test_dataloders[index])
        testing_corrects += corrects
        testing_sum += num
    print(f"Acc: {testing_corrects / testing_sum}")
    return (testing_corrects / testing_sum)


def get_sub_ViT_module_name(model):
    # for ViT model
    name_list = []
    for i, _ in model.named_modules():
        if i == "model.conv_proj" or i == "model.encoder.ln" or i == "model.heads.head":
            name_list.append(i)
        elif len(i.split(".")) > 4 and "dropout" not in i:    #and "dropout" not in i
            name_list.append(i)
    return name_list

def get_sub_ResNet_module_name(model):
    name_list = []
    for i, _ in model.named_modules():
        if i in ["backbone.conv1", "backbone.bn1", "backbone.relu", "backbone.maxpool", "backbone.avgpool", "backbone.fc"]:
            name_list.append(i)
        elif len(i.split(".")) >= 4:
            name_list.append(i)
    return name_list

def get_top_VIT_module_name(model):
    name_list = []
    for i, _ in model.named_modules():
        if len(i.split(".")) <= 4:
            if i not in ["", "model", "model.encoder.dropout", "model.heads.head", "model.encoder.ln","model.encoder.layers"]:
                name_list.append(i)
        elif i.split(".")[-1] == "mlp":
            name_list.append(i)
    return name_list


def compute_st_bias(prob, gt_label):
    gt_label = gt_label.detach().cpu()
    prob = prob.detach().cpu()
    bias = 0
    for i in range(prob.shape[0]):
        bias += prob[i][gt_label[i]]

    return bias / prob.shape[0]

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')



class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.model = vit_b_16(weights= ViT_B_16_Weights.IMAGENET1K_V1)
        self.heads_layers = OrderedDict()
        self.heads_layers['head'] = nn.Linear(768, num_classes)
        self.model.heads = nn.Sequential(self.heads_layers)

    def forward(self, x):
        logits = self.model(x)
        return logits, softmax(logits, dim=-1)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights = ResNet50_Weights)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)
    def forward(self, x):
        logits = self.backbone(x)
        return logits, softmax(logits, dim=-1)



def generate_bar_chart(dictionary, save_directory):
    # Extract keys and values from the dictionary
    # layers = list(dictionary.keys())
    values = list(dictionary.values())
    layers = [".".join(x.split(".")[-4:]) for x in dictionary.keys() ]


    # Create a bar chart
    plt.bar(layers, values, color='blue')
    plt.xlabel('Model Layer')
    plt.ylabel('Number')
    plt.title('Total Effect of Model Layers')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

    # Annotate each bar with its value
    # for i, value in enumerate(values):
    #     plt.text(i, value + 0.1, str(value), ha='center', va='bottom')

    # Save the bar chart as an image file
    plt.savefig(save_directory)
    # plt.show()

# def get_better_bias(local_bias, global_bias, mod_bias, ):
#     # compare the higher



if __name__=="__main__":

    model = ResNet50(num_classes=4)
    print()
    n_list = get_sub_ResNet_module_name(model)
    print(n_list)
    # model_1 = ViT(num_classes=4)
    # i = get_top_module_name(model)
    # # module_name = 'model.encoder.layers.encoder_layer_0.mlp.0'
    # block_name = 'model.heads'
    # print(model.state_dict()['model.heads.head.weight'])
    # print(model_1.state_dict()['model.heads.head.weight'])
    # module = None
    # for m,n in model_1.named_modules():
    #     if m==block_name: module = n
    # setattr(model, block_name, module)
    # i = get_top_module_name(model)
    # print(len(i))
    # print(model.state_dict()['model.heads.head.weight'])





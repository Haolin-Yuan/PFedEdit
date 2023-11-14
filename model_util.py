import torch.nn as nn
from torch.nn.functional import softmax
import torch
from torch.nn.functional import relu
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights, resnet18, ResNet18_Weights
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import copy
import torchvision


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

def new_evaluate_global(users, test_dataloders, NUM_USER):
    testing_corrects = 0
    testing_sum = 0
    for index in range(NUM_USER):
        for i in range(NUM_USER):
            corrects, num = users[index].evaluate(test_dataloders[i])
            testing_corrects += corrects
            testing_sum += num
    print(f"All Acc: {testing_corrects / testing_sum}")
    return testing_corrects / testing_sum


def get_sub_ViT_module_name(model):
    # for ViT model
    name_list = []
    for i, _ in model.named_modules():
        if i == "model.conv_proj" or i == "model.encoder.ln" or i == "model.heads.head":
            name_list.append(i)
        elif len(i.split(".")) > 4 and "dropout" not in i:    #and "dropout" not in i
            if i.split(".")[-1]!= "mlp":name_list.append(i)
    return name_list

def get_sub_ResNet_module_name(model):
    name_list = []
    for i, _ in model.named_modules():
        if len(i.split(".")) < 4:
            if i in ["backbone.avgpool", "backbone.conv1", "backbone.bn1", "backbone.relu", "backbone.maxpool"]:
                name_list.append(i)
        else:
            name_list.append(i)
    # name_list.remove('')
    # name_list.remove('backbone')
    return name_list

def get_sub_VGG_module_name(model):
    name_list = []
    for i,_ in model.named_modules():
        if i not in ["", "network", "linear_layers"]:
            name_list.append(i)
    return name_list

def get_MLP_module_name(model):
    name_list = []
    for x, _ in model.named_modules():
        if x != "": name_list.append(x)
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

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet18(weights = ResNet18_Weights)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)
    def forward(self, x):
        logits = self.backbone(x)
        return logits, softmax(logits, dim=-1)


class MLP(nn.Module):
    def __init__(self,num_classes):
        super(MLP, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.hidden3 = 64
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=False)
        self.fc4 = nn.Linear(self.hidden3, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x = x.view(-1, 28 * 28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        logits = self.fc4(x)
        return logits, softmax(logits,dim=1)


class VGG_11(nn.Module):
    def __init__(self, num_classes):
        super(VGG_11, self).__init__()
        self.network = torchvision.models.vgg11(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        logits = self.linear_layers(x)
        return logits, softmax(logits,dim=1)



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

def get_model_list(model_name, model):
    if model_name == "ViT":
        module_name_list = get_sub_ViT_module_name(model)
    elif model_name == "ResNet18":
        module_name_list = get_sub_ResNet_module_name(model)
    elif model_name == "MLP":
        module_name_list = get_MLP_module_name(model)
    elif model_name == "VGG_11":
        module_name_list = get_sub_VGG_module_name(model)
    return module_name_list

if __name__=="__main__":

    model = ResNet18(4)
    print(model)
    x = get_sub_ResNet_module_name(model)
    print(x)
    # print(len(x))





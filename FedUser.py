import torch
import torchmetrics
import numpy as np
from model_util import *
from tqdm import tqdm
import torch.nn as nn
from ResNet import ResNet_cifar
import copy
import math



class FedUser():
    def __init__(self, index, device, model, n_classes, train_dataloader, epochs, lr, num_layer, module_name_list):
        self.model_name = model
        self.index = index
        self.device = device
        self.model = globals()[model](num_classes = n_classes)
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes).to(device)
        # decide which layer should be the private layer to be edited locally
        self.layer_name = []
        self.hook_list = []
        self.model_hook = {}
        self.clean_hooks = []
        self.previous_iter_model_weight = globals()[model](num_classes = n_classes)
        self.num_layer = num_layer
        self.module_name_list = module_name_list



    def train(self):
        self.model.to(self.device)
        self.model.train()
        for e in range(self.epochs):
            losses = []
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits, pred = self.model(x)
                pred = torch.argmax(pred, dim= 1)       # for ciafar dataset only
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
                self.acc_metric(pred, y)
                losses.append(loss.item())
            print(f"Client: {self.index}, ACC: {self.acc_metric.compute()}, Loss: {np.mean(losses)}")
            self.acc_metric.reset()
        self.model.to("cpu")


    def evaluate(self, dataloader):
        self.model.to(self.device)
        self.model.eval()
        testing_corrects = 0
        testing_sum = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                _, preds = self.model(x)
                testing_corrects += torch.sum(torch.argmax(preds, dim=1) == y)
                testing_sum += len(y)
        self.model.to("cpu")
        return testing_corrects.cpu().detach(), testing_sum


    def forward_hook(self,module_name):
        def hook(module, input, output):
            self.model_hook[module_name] = output
            # for i in self.model_hook[module_name]:
            #     if i != None: i.detach().cpu()
        return hook

    def recover_hidden_state_value(self, module_name, index):
        def modified_hook(module, input, output):
            # clone will back propagate!!!
            self.model_hook[module_name].copy_(self.clean_hooks[index][module_name])
            return self.model_hook[module_name]
        return modified_hook

    def register(self, model):
        for name, module in model.named_modules():
            module.register_forward_hook(self.forward_hook(name))


    def get_module(self, name):
        for n,m in self.previous_iter_model_weight.named_modules():
            if n == name:return m.to(self.device)      #same if using copy.deepcopy() or not
        raise LookupError(name)

    def recover_from_clean_model(self, model, module_name):
        module = self.get_module(module_name)
        setattr(model, module_name, module)

        return model



    def eval_model_with_hook(self, model, test_loader, bias, recover, recovered_name=None):
        model.to(self.device)
        model.eval()

        if recover:
            for name, module in self.previous_iter_model_weight.named_modules():
                if recovered_name in name:
                    setattr(model, name, module.to(self.device))
            gt_match_list = []

        # losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, y_pred = model(x)
                # loss = self.loss_fn(logits, y)
                # losses.append(loss.item())
                bias.append(compute_st_bias(y_pred, y))
                if recover:
                    for a, b in zip(y_pred, y): gt_match_list.append(True) if torch.argmax(a) == b else gt_match_list.append(False)
                # hook_list.append(self.model_hook)
                del y_pred, logits
        model.to("cpu")
        if recover:
            return bias, gt_match_list
        else: return bias

    def compare_hooks(self, hook_1, hook_2):
        models_differ = 0
        mismatch = []
        for key in hook_1.keys():
            for item_1, item_2 in zip(hook_1[key], hook_2[key]):
                if item_1 != None and item_2 != None:
                    if torch.equal(item_1, item_2):
                        pass
                    else:
                        models_differ += 1
                        mismatch.append(key)
        if models_differ == 0:
            print('Hooks match perfectly! :)')
        else:
            print('difference found!', mismatch)
            print(f'mismatch ratio: {(len(mismatch) / len(hook_1.keys())) * 100}%')


    def casual_trace(self, data_loader):
        '''
        compuet the total effect of recovering each layer
        :return:
        '''
        self.layer_name = []
        total_effect = {}
        clean_local_bias  = self.eval_model_with_hook(model=self.previous_iter_model_weight,
                                                            test_loader= data_loader,recover=False,
                                                            bias=[])

        print(self.model_name, len(self.module_name_list), "# replaced layer ",self.num_layer)

        for i, _ in self.model.named_modules():
            if i in self.module_name_list:

                recovered_bias, gt_match_list = self.eval_model_with_hook(model= copy.deepcopy(self.model),
                                                    test_loader= data_loader, recover=True,
                                                    bias=[], recovered_name=i)



                # original metrics
                total_effect[i] = [recovered_bias[x] / clean_local_bias[x] - 1 for x in range(len(clean_local_bias))]

                # new metrics
                tf_list = {"A": 0, "B":0, "C":0, "D":0, "E":0}

                # A mask, val >0, B mask > 0, val <0; C mask<0 , val >0, D, mask, val<0
                for val, mask in zip(total_effect[i], gt_match_list):
                    if mask and val >0:
                        tf_list["A"] += 1
                    elif mask and val < 0:
                        tf_list["B"] += 1
                    elif not mask and val >0:
                        tf_list["C"] += 1
                    elif not mask and val < 0:
                        tf_list["E"] += 1
                tf_list["D"] += sum(total_effect[i])/ len(clean_local_bias)
                total_effect[i] = tf_list

                del recovered_bias

        def custom_sort(data):
            return (data[1]["A"], data[1]["B"], data[1]["C"], data[1]["D"], data[1]["E"])

        total_effect = sorted(total_effect.items(), key=custom_sort, reverse=True)


        print(total_effect)

        for i in range(self.num_layer):
            val, _ = total_effect[i]
            self.layer_name.append(val)



    def get_model_state_dict(self):
        return self.model.state_dict()


    def set_previous_local_weights(self):
        # record previous local training weights
        for key, value in self.model.state_dict().items():
            self.previous_iter_model_weight.state_dict()[key].data.copy_(self.model.state_dict()[key])



    def set_model_state_dict(self, weights):
        # self.set_previous_local_weights()
        if self.layer_name == []:
            for key, value in self.model.state_dict().items():
                    self.model.state_dict()[key].data.copy_(weights[key])
        else:
            for key, value in self.model.state_dict().items():
                for i in range(len(self.layer_name)):
                    if self.layer_name[i] not in key:
                        self.model.state_dict()[key].data.copy_(weights[key])



if __name__ == "__main__":
    # model = ResNet_cifar(num_classes=10)
    model = "ResNet_cifar"
    model = globals()[model](num_classes = 10)
    print(model)

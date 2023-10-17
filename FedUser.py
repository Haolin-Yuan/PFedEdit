import torch
import torchmetrics
import numpy as np
from model_util import *
from tqdm import tqdm
import torch.nn as nn
import copy



class FedUser():
    def __init__(self, index, device, model, n_classes, train_dataloader, epochs, lr, num_layer):
        self.index = index
        self.device = device
        self.model = globals()[model](num_classes = n_classes)
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(device)
        # decide which layer should be the private layer to be edited locally
        self.layer_name = []

        self.hook_list = []
        self.model_hook = {}
        # clean hook list for iid samples
        self.clean_hooks = []
        # corrupted hook list for non-iid samples
        # self.recovered_hooks = []
        self.previous_iter_model_weight = globals()[model](num_classes = n_classes)
        # self.MODEL_PATH = f"weights/local/{index}.pt"
        self.num_layer = num_layer

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for e in range(self.epochs):
            losses = []
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # one_hot_y = torch.nn.functional.one_hot(y, num_classes=10)
                self.optimizer.zero_grad()
                logits, pred = self.model(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
                self.acc_metric(pred, y)
                losses.append(loss.item())
            print(f"Client: {self.index}, ACC: {self.acc_metric.compute()}, Loss: {np.mean(losses)}")
            self.acc_metric.reset()
        # torch.save(self.model.state_dict(), self.MODEL_PATH)
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
            # self.model_hook[module_name] = torch.ones(self.model_hook[module_name].shape).to(self.device)
            # self.model_hook[module_name] = self.clean_hooks[index][module_name].clone()
            # clone will back propagate!!!
            self.model_hook[module_name].copy_(self.clean_hooks[index][module_name])
            # if isinstance(self.model_hook[module_name], tuple):
            #     return (self.model_hook[module_name][0].to(self.device), self.model_hook[module_name][1])
            # else:
            #     return self.model_hook[module_name].to(self.device)
            return self.model_hook[module_name]
        return modified_hook

    def register(self, model):
        for name, module in model.named_modules():
            module.register_forward_hook(self.forward_hook(name))


    def get_module(self, name):
        for n,m in self.model.named_modules():
            if n == name: return m
        raise LookupError(name)

    def recover_from_clean_model(self, model, module_name):
        module = self.get_module(module_name)
        setattr(model, module_name, module)
        return model



    def eval_model_with_hook(self, model, test_loader, hook_list, bias, recover, recovered_name=None):
        # model = nn.DataParallel(model, device_ids =[0, 1, 2, 3])
        model.to(self.device)
        model.eval()
        # recover_count = 0


        if recover:
            dummpy_model = copy.deepcopy(model)
            for name, module in dummpy_model.named_modules():
                if recovered_name in name:
                    model = self.recover_from_clean_model(dummpy_model, name)
            del dummpy_model

        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(self.device), y.to(self.device)

                # for name, module in model.named_modules():
                #     if recover == True and recovered_name in name:
                #         module.register_forward_hook(self.recover_hidden_state_value(name, index= len(hook_list)))
                #         # print(f"After change, model hook: {self.model_hook[name]}")
                #     else:
                #         module.register_forward_hook(self.forward_hook(name))

                logits, y_pred = model(x)
                # if recover:
                #     print("====="*30)
                #     self.compare_hooks(self.model_hook, self.clean_hooks[len(hook_list)])
                #     print("=====" * 30)
                bias.append(compute_st_bias(y_pred, y))
                hook_list.append(self.model_hook)
                # ModelForwardHook.collect_clean_hook() if not recovered_name else ModelForwardHook.collect_corrupted_hook()
                # if not recovered_name: hook_list.append(self.model_hook.copy())

                # for key in self.model_hook.keys():
                #     if isinstance(self.model_hook[key], tuple):
                #         self.model_hook[key] = (self.model_hook[key][0].detach().cpu(), self.model_hook[key][1])
                #     elif self.model_hook[key] != None:
                #         self.model_hook[key] = self.model_hook[key].detach().cpu()
                # test_loss = loss_fn(logits, y)
                # total_loss += test_loss.item()
                # total_correct_num += torch.sum(torch.argmax(y_pred, dim=1) == y)
                # test_sum += len(y)
                # del x, y
                del y_pred, logits
        model.to("cpu")
        # bias_sum = sum(bias)
        # del bias
        # if recover: hook_list = []
        return hook_list, bias

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

    def compare_models(self, model_1, model_2):
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

    def casual_trace(self, data_loader):
        '''
        compuet the total effect of recovering each layer
        :return:
        '''
        total_effect = {}

        # self.compare_models(self.model, self.previous_iter_model_weight)
        self.clean_hooks, clean_bias = self.eval_model_with_hook(model=self.model, test_loader= data_loader,recover=False,
                                                            hook_list=self.clean_hooks, bias=[])

        # print("clean bias", clean_bias)
        for i, _ in self.model.named_modules():
            if ".mlp.0" in i or ".mlp.3" in i:
                recovered_hooks, recovered_bias = self.eval_model_with_hook(model=self.previous_iter_model_weight,
                                                            test_loader= data_loader, recover=True,
                                                            hook_list= [],
                                                            bias=[], recovered_name=i)
                # print("recover bias", recovered_bias)
                # for i in range(len(recovered_hooks)):
                #     self.compare_hooks(self.clean_hooks[i], recovered_hooks[i])

                # total_effect[i] = sum(recovered_bias) / sum(clean_bias) - 1  # total effect
                total_effect[i] = sum([recovered_bias[x] / clean_bias[x] - 1 for x in range(len(clean_bias))])  # total effect
                del recovered_bias

        # to be done: sort importance from high to low
        total_effect = dict(sorted(total_effect.items(), key=lambda x: x[1], reverse=True))
        print(total_effect)
        print("-----"*30)

        for i in range(self.num_layer):
            layer = max(total_effect, key=lambda x: total_effect[x])
            self.layer_name.append(layer)
            total_effect.pop(layer)

        print(f"layer name for user {self.index}: {self.layer_name}")
        del total_effect



    def get_model_state_dict(self):
        return self.model.state_dict()

    def set_previous_local_weights(self):
        # record previous local training weights
        for key, value in self.model.state_dict().items():
            self.previous_iter_model_weight.state_dict()[key].data.copy_(self.model.state_dict()[key])
        print("previous model weights set")


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




# from datasets import *
# from model_util import *
# from torch.utils.data import DataLoader
# _, test_set = get_datasets("lyme", dataroot= None)
# test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=0)
# user = FedUser(index = 0, device=torch.device("cuda:0"), model = "ViT",
#                n_classes = 4, train_dataloader= test_loader,epochs = 10, lr=1e-3)
#
# hook_list = []
# bias = []
# hook_list, bias = user.eval_model_with_hook(test_loader, hook_list, bias, recover_clean=False)
# print(len(hook_list))

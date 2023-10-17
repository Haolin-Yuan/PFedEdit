import torch
from model_util import *

class FedServer():

    def __init__(self, device, model, sample_clients, n_classes):
        self.model = globals()[model](num_classes = n_classes)
        self.device = device
        self.sample_clients = sample_clients
        self.trainable_names = [k for k,_ in self.model.named_parameters()]
        # self.MODEL_PATH = "weights/server/server.pt"


    def get_model_state_dict(self):
        return self.model.state_dict()

    def agg_update(self, weights):
        with torch.no_grad():
            print("aggregating------------")
            for k, v in self.get_model_state_dict().items():
                sumed_grad = torch.zeros_like(v)
                for i in range(len(weights)):
                    grad = weights[i][k] - v
                    sumed_grad += grad
                value = v + sumed_grad/self.sample_clients
                self.model.state_dict()[k].data.copy_(value.detach().clone())

    def agg_update_with_index(self, weights, layer_name):
        with torch.no_grad():
            print("aggregating with each index ------------")

            for k, v in self.get_model_state_dict().items():
                sumed_grad = torch.zeros_like(v)
                if not layer_name:
                    for i in range(len(weights)):
                        grad = weights[i][k] - v
                        sumed_grad += grad
                    value = v + sumed_grad / self.sample_clients
                else:
                    agg_count = 0
                    for i in range(len(weights)):
                        for j in range(len(layer_name[i])):
                            if layer_name[i][j] not in k:
                                grad = weights[i][k] - v
                                sumed_grad += grad
                                agg_count += 1
                        # print(f"k: {k}, layer name: {layer_name[i]}, agg count: {agg_count}")
                    value = v + sumed_grad / agg_count

                self.model.state_dict()[k].data.copy_(value.detach().clone())




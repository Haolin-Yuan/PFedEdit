import torch
from model_util import *
from FedUser import *
from FedServer import *
from datasets import *
import numpy as np


ROUND = 10
NUM_USERS = 10
NUM_CLASSES = 4
MODEL = "ViT"
LOCAL_EPOCHS = 1
DATA_NAME = "lyme"
root = None
BATCH_SIZE = 1
NUM_CLASS_CLIENT = 2
SAMPLE_RATE = 1
LR = 1e-3

# Layer_name = "model.encoder.layers.encoder_layer_11.mlp.0"

device = torch.device("cuda:0")
train_loader, test_loader = gen_random_loaders(DATA_NAME, root,
                                               NUM_USERS, BATCH_SIZE, NUM_CLASS_CLIENT, NUM_CLASSES)
fed_users = [FedUser(i, device, MODEL, NUM_CLASSES, train_loader[i], LOCAL_EPOCHS, LR) for i in range(NUM_USERS)]
fed_server = FedServer(device, MODEL, int(SAMPLE_RATE*NUM_USERS), NUM_CLASSES)

for i in range(NUM_USERS):
    fed_users[i].set_model_state_dict(fed_server.get_model_state_dict())
for round in range(1):
    rand_index = np.random.choice(NUM_USERS, int(SAMPLE_RATE*NUM_USERS), replace =False)
    for index in rand_index:
        fed_users[index].train()
    fed_server.agg_update([fed_users[index].get_model_state_dict() for index in rand_index])
# for i in range(1,NUM_USERS):
#     del fed_users[1]
del train_loader
for i in range(NUM_USERS):
    Server_PATH = f"weights/server/server.pt"
    MODEL_PATH = f"weights/local/{i}.pt"
    fed_users[i].model = ViT(num_classes = NUM_CLASSES)
    fed_users[i].model.load_state_dict(torch.load(Server_PATH))
    fed_users[i].previous_iter_model_weight = ViT(num_classes=NUM_CLASSES)
    fed_users[i].previous_iter_model_weight.load_state_dict(torch.load(MODEL_PATH))
    # fed_users[i].set_model_state_dict(fed_server.get_model_state_dict())
    print(f"Starting casual trace : user {i}--------------------------------------------")
    fed_users[i].casual_trace(test_loader[i])
    fed_users[i].model = None
    fed_users[i].clean_hooks = []
    fed_users[i].previous_iter_model_weight =None

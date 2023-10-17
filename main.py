import torch
from model_util import *
from FedUser import *
from FedServer import *
from datasets import *
import numpy as np
from model_editing import apply_model_editing

ROUND = 4
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
NUM_LAYER = 1   # numbers of layers we edit
Vanilla = False
customized_Agg = True

device = torch.device("cuda:0")
train_loader, val_loader, test_loader = gen_random_loaders(DATA_NAME, root,
                                               NUM_USERS, BATCH_SIZE, NUM_CLASS_CLIENT, NUM_CLASSES)
fed_users = [FedUser(i, device, MODEL, NUM_CLASSES, train_loader[i], LOCAL_EPOCHS, LR, NUM_LAYER) for i in range(NUM_USERS)]
fed_server = FedServer(device, MODEL, int(SAMPLE_RATE*NUM_USERS), NUM_CLASSES)

# def start_casual_trace():
#     for i in range(NUM_USERS):
#         fed_users[i].model = None
#         fed_users[i].previous_iter_model_weight = None
#
#     for i in range(NUM_USERS):
#         Server_PATH = f"weights/server/server.pt"
#         MODEL_PATH = f"weights/local/{i}.pt"
#         fed_users[i].model = ViT(num_classes=NUM_CLASSES)
#         fed_users[i].model.load_state_dict(torch.load(Server_PATH))
#         fed_users[i].previous_iter_model_weight = ViT(num_classes=NUM_CLASSES)
#         fed_users[i].previous_iter_model_weight.load_state_dict(torch.load(MODEL_PATH))
#         # fed_users[i].set_model_state_dict(fed_server.get_model_state_dict())
#         print(f"Starting casual trace : user {i}--------------------------------------------")
#         fed_users[i].casual_trace(test_loader[i])
#         fed_users[i].model = None
#         fed_users[i].clean_hooks = []
#         fed_users[i].previous_iter_model_weight = None
#     print("Casual Trace Done --------------------------------------------")
#     for i in range(NUM_USERS):
#         Server_PATH = f"weights/server/server.pt"
#         MODEL_PATH = f"weights/local/{i}.pt"
#         fed_users[i].model = ViT(num_classes=NUM_CLASSES)
#         fed_users[i].model.load_state_dict(torch.load(Server_PATH))
#         fed_users[i].previous_iter_model_weight = ViT(num_classes=NUM_CLASSES)
#         fed_users[i].previous_iter_model_weight.load_state_dict(torch.load(MODEL_PATH))


for i in range(NUM_USERS):
    fed_users[i].set_model_state_dict(fed_server.get_model_state_dict())
best_acc = 0
for round in range(ROUND):
    # rand_index = np.random.choice(NUM_USERS, int(SAMPLE_RATE*NUM_USERS), replace =False)
    for i in range(NUM_USERS):
        # if round == 0:
        #     fed_users[index].epochs = 5
        # else:
        #     fed_users[index].epochs = LOCAL_EPOCHS
        fed_users[i].train()

    if Vanilla:
        fed_server.agg_update([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)])
    else:
        if customized_Agg == True:
            # aggregate model weights depends on if there are private layers
            if round == 0:
                fed_server.agg_update([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)])
            else:
                for i in range(NUM_USERS): print(fed_users[i].index, fed_users[i].layer_name)
                agg_layer_name = {fed_users[i].index: fed_users[i].layer_name for i in range(NUM_USERS)}
                fed_server.agg_update_with_index([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)], agg_layer_name)
        else:
            fed_server.agg_update([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)])

    for i in range(NUM_USERS):
        fed_users[i].set_previous_local_weights()
        fed_users[i].set_model_state_dict(fed_server.get_model_state_dict())

    if Vanilla == False and round == 0:
        for i in range(NUM_USERS): fed_users[i].casual_trace(val_loader[i])

    print(f"Round: {round + 1}")
    acc = evaluate_global(fed_users, test_loader, range(NUM_USERS))
    if acc > best_acc:
        best_acc = acc
# start model editing
if Vanilla == False:
    # for i in range(NUM_USERS): fed_users[i].casual_trace(test_loader[i])
    print("Start model editing ----------------------------------")
    for i in range(NUM_USERS):
        for k in range(NUM_LAYER):
            print("Editing layer # ", k)
            fed_users[i].model = apply_model_editing(fed_users[i].model,fed_users[i].previous_iter_model_weight,
                                                    fed_users[i].layer_name[k], val_loader[i], fed_users[i].device)

    acc = evaluate_global(fed_users, test_loader, range(NUM_USERS))

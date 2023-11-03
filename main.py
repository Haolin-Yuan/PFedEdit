import torch
from model_util import *
from FedUser import *
from FedServer import *
from datasets import *
import numpy as np
from model_editing import apply_model_editing, replace_model_layer

ROUND = 10
NUM_USERS = 10
NUM_CLASSES = 4
MODEL = "ResNet50"
LOCAL_EPOCHS = 1
DATA_NAME = "lyme"
root = None
BATCH_SIZE = 1
NUM_CLASS_CLIENT = 2
SAMPLE_RATE = 1
LR = 1e-3
Agg_all = True
Vanilla = False
num_replace_layer = 53

device = torch.device("cuda:0")

train_loader, val_loader, test_loader = gen_random_loaders(DATA_NAME, root,
                                               NUM_USERS, BATCH_SIZE, NUM_CLASS_CLIENT, NUM_CLASSES)
fed_users = [FedUser(i, device, MODEL, NUM_CLASSES, train_loader[i], LOCAL_EPOCHS, LR, num_replace_layer) for i in range(NUM_USERS)]
fed_server = FedServer(device, MODEL, int(SAMPLE_RATE*NUM_USERS), NUM_CLASSES)
# top_module_list = get_sub_module_name(fed_users[0].model)

for i in range(NUM_USERS):
    fed_users[i].set_model_state_dict(fed_server.get_model_state_dict())

best_acc = 0
for round in range(ROUND):
    for i in range(NUM_USERS):
        fed_users[i].train()

    if  Agg_all == True:
        weights_agg = agg_weights([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)])
    else:
        # aggregate model weights depends on if there are private layers
        if round == 0:
            weights_agg = agg_weights([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)])
        else:
            for i in range(NUM_USERS): print(fed_users[i].index, fed_users[i].layer_name)
            agg_layer_name = {fed_users[i].index: fed_users[i].layer_name for i in range(NUM_USERS)}
            weights_agg = custom_agg_weights([fed_users[index].get_model_state_dict() for index in range(NUM_USERS)], agg_layer_name)

    for i in range(NUM_USERS):
        print(f"Round :{round+1}, user {i}")
        fed_users[i].set_previous_local_weights()
        fed_users[i].set_model_state_dict(weights_agg)

        if not Vanilla:
            fed_users[i].casual_trace(val_loader[i])
            for k in range(num_replace_layer):
                print(f" round {round+1} user {i} repalced layer: {fed_users[i].layer_name[k]}")
                fed_users[i].model = fed_users[i].recover_from_clean_model(fed_users[i].model, fed_users[i].layer_name[k])

        # fed_users[i].top_module_name_list = top_module_list.copy()
        # print(f"user {i}", fed_users[i].top_module_name_list)
        # for k in range(num_replace_layer):
        #     print(f"user {i}, {k} round casual trace")
        #     fed_users[i].casual_trace(val_loader[i])
        #     if fed_users[i].layer_name != []:
        #         fed_users[i].model = fed_users[i].recover_from_clean_model(fed_users[i].model, fed_users[i].layer_name[0])
        #     else: print("----"*10)
        #     print(f"user {i}, {k} round casual trace, no layer replaced")


        # merge m_global with previous local model
        # print(f"now setting averaged model weights for user {i}")
        # averaged_weights  = merge_two_model_weights(fed_users[i].get_model_state_dict(),
        #                                              fed_users[i].previous_iter_model_weight.state_dict())
        # fed_users[i].set_model_state_dict(averaged_weights)

    print(f"Round: {round + 1}")
    acc = evaluate_global(fed_users, test_loader, range(NUM_USERS))
    if acc > best_acc:
        best_acc = acc
print(f"Best Acc: {best_acc}")

# start model editing
# print("Start model editing ----------------------------------")
# for i in range(NUM_USERS):
#     for k in range(NUM_LAYER):
#         print("Editing layer # ", k)
#         fed_users[i].model = apply_model_editing(fed_users[i].model,fed_users[i].previous_iter_model_weight,
#                                                 fed_users[i].layer_name[k], val_loader[i], fed_users[i].device)
#
# acc = evaluate_global(fed_users, test_loader, range(NUM_USERS))


# if __name__=="__main__":
#     model = ViT(num_classes=4)
#     train_loader, val_loader, test_loader = gen_random_loaders(DATA_NAME, root,
#                                                                NUM_USERS, BATCH_SIZE, NUM_CLASS_CLIENT, NUM_CLASSES)
#     model = apply_model_editing(model, model, "model.encoder.layers.encoder_layer_0.mlp.3", val_loader[0], device)
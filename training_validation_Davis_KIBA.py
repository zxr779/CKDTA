import sys
import torch.nn as nn

import torch_geometric.transforms as T
from utils import *
from lifelines.utils import concordance_index
from create_data import create_dataset
import time
from torch.optim.lr_scheduler import LambdaLR
from models.CKDTA import CKDTA
import pickle
from setrandseed import *

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.002
NUM_EPOCHS = 2000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

datasets = ['filter_davis']
modeling = [CKDTA][0]
model_st = modeling.__name__

print("dataset:", datasets)
print("modeling:", modeling)

# determine the device in the following line
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)



# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)

    start_time = time.time()
    train_data_file = "./data/" + dataset + "_traindata_"+ str(TRAIN_BATCH_SIZE) +".data"
    test_data_file = "./data/" + dataset + "_testdata_"+ str(TEST_BATCH_SIZE) +".data"
    if not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file)):
        train_data, test_data = create_dataset(dataset)
        print(train_data[0])
        torch.save(train_data, train_data_file)  # save train data
        torch.save(test_data, test_data_file)  # save test data
    else:
        train_data = torch.load(train_data_file)
        test_data = torch.load(test_data_file)

    print('load dataset successfully')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    print('complete dataloader loading')
    end_time = time.time()
    all_time = end_time-start_time
    print('The data preparation took a total of ',all_time,' seconds')
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lambda_lr = lambda epoch: 0.5 ** (epoch // 1000)
    # 初始化调度器
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    best_mse = 400
    best_ci = 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_' + dataset +  'ca.pth'
    result_file_name = 'result_' + model_st + '_' + dataset +  'ca.csv'

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for epoch in range(NUM_EPOCHS):

        epoch_start_time = time.time()  # 记录每轮开始的时间

        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = evaluate(model, device, test_loader, loss_fn)
        G, P = predicting(model, device, test_loader)
        ret = [mse(G, P), concordance_index(G, P)]

        epoch_end_time = time.time()  
        epoch_time = epoch_end_time - epoch_start_time  

        print(f'Epoch {epoch + 1} took {epoch_time:.2f} seconds')

        scheduler.step()

        if ret[0] < best_mse:
            torch.save(model, model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch+1
            best_mse = ret[0]
            best_ci = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
        elif (epoch - best_epoch)<1000:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
        else:
            print('early stop  ''; best_mse,best_ci:', best_mse, best_ci,model_st, dataset)
            break
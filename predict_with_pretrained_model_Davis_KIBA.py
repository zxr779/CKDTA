from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from utils import *
from create_data import create_dataset
import time
import seaborn as sns
from models.CKDTA import CKDTA
import numpy as np
datasets = ['filter_davis']
modelings = [CKDTA]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE =256

result = []
for dataset in datasets:

    start_time = time.time()
    # load data
    test_data_file = "./data/" + dataset + "_testdata_"+str(TEST_BATCH_SIZE)+".data"
    if not os.path.isfile(test_data_file):
        _, test_data = create_dataset(dataset)
        torch.save(test_data, test_data_file)
    else:
        test_data = torch.load(test_data_file)


    print('load dataset successfully')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print('Complete dataloader loading')
    end_time = time.time()
    all_time = end_time - start_time
    print('The data preparation took a total of ', all_time, ' seconds')
    # Predicting the affinity of test data
    for modeling in modelings:
        model_st = modeling.__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)

        model_file_name = 'model_CKDTA_filter_davisca.pth'
        if os.path.isfile(model_file_name):
            model = torch.load(model_file_name)
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P),rm2(G,P)]
            ret =[dataset, model_st] + [round(e, 3) for e in ret]
            result += [ ret ]
            print('dataset,model,rmse,mse,pearson,spearman,ci,rm2')
            print(ret)
        else:
            print('model is not available!')

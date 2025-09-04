import sys
import torch.nn as nn
import torch_geometric.transforms as T
from utils import *
from lifelines.utils import concordance_index
from create_dataset_filter import create_dataset
import time
from torch.optim.lr_scheduler import LambdaLR
from models.CKDTA import CKDTA
from randseed import *
from dataloadering import *
import pickle
from sklearn.model_selection import KFold
import pandas as pd

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
seed_everything()
LR = 0.002
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
N_SPLITS = 5  # 五折交叉验证

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Cross-validation folds: ', N_SPLITS)

datasets = ['filter davis']
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
    print('\nrunning on ', model_st + '_' + dataset + ' with 5-fold cross-validation')

    start_time = time.time()

    # 加载完整数据集
    full_data_file = "./data/" + dataset + "_fulldata.data"
    if not os.path.isfile(full_data_file):
        # 如果没有完整数据集文件，则创建
        train_data, test_data = create_dataset('davis')
        # 合并训练和测试数据作为完整数据集
        full_data = train_data + test_data
        torch.save(full_data, full_data_file)
        print('Created full dataset and saved to file')
    else:
        full_data = torch.load(full_data_file)
        print('Loaded full dataset from file')

    print('Full dataset size:', len(full_data))

    # 准备存储每折结果
    fold_results = []
    all_fold_metrics = []

    # 设置K折交叉验证
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 开始K折交叉验证
    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_data)):
        print(f'\n{"=" * 50}')
        print(f'FOLD {fold + 1}/{N_SPLITS}')
        print(f'{"=" * 50}')

        # 创建当前折的训练和验证数据
        train_data_fold = [full_data[i] for i in train_indices]
        val_data_fold = [full_data[i] for i in val_indices]

        print(f'Training samples: {len(train_data_fold)}')
        print(f'Validation samples: {len(val_data_fold)}')

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_data_fold,
                                                   batch_size=TRAIN_BATCH_SIZE,
                                                   shuffle=True,
                                                   collate_fn=collate)
        val_loader = torch.utils.data.DataLoader(val_data_fold,
                                                 batch_size=TEST_BATCH_SIZE,
                                                 shuffle=False,
                                                 collate_fn=collate)

        # 初始化模型
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        lambda_lr = lambda epoch: 0.95 ** (epoch // 500)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

        # 为当前折创建日志文件
        loss_log_file = open(f'loss_{dataset}_fold_{fold + 1}.txt', 'w')

        # 训练参数
        best_mse = 400
        best_ci = 0
        best_epoch = -1
        model_file_name = f'model_{model_st}_{dataset}_fold_{fold + 1}.pth'
        result_file_name = f'result_{model_st}_{dataset}_fold_{fold + 1}.csv'

        # 打印模型参数
        if fold == 0:  # 只在第一折打印一次
            total_params = sum(p.numel() for p in model.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

        fold_start_time = time.time()

        # 训练循环
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()

            train_loss = train(model, device, train_loader, optimizer, epoch)
            val_loss = evaluate(model, device, val_loader, loss_fn)
            G, P = predicting(model, device, val_loader)
            ret = [mse(G, P), concordance_index(G, P)]

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            print(f'Fold {fold + 1}, Epoch {epoch + 1} took {epoch_time:.2f} seconds')

            loss_log_file.write(f'Epoch {epoch + 1}: Training Loss: {train_loss}, Validation Loss: {val_loss}\n')
            loss_log_file.flush()

            scheduler.step()

            if ret[0] < best_mse:
                torch.save(model, model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[0]
                best_ci = ret[-1]
                print(
                    f'Fold {fold + 1}: MSE improved at epoch {best_epoch}; best_mse: {best_mse:.4f}, best_ci: {best_ci:.4f}')
            elif (epoch - best_epoch) < 100:  # 减少早停的等待轮次
                print(
                    f'Fold {fold + 1}: MSE: {ret[0]:.4f}, No improvement since epoch {best_epoch}; best_mse: {best_mse:.4f}, best_ci: {best_ci:.4f}')
            else:
                print(f'Fold {fold + 1}: Early stop; best_mse: {best_mse:.4f}, best_ci: {best_ci:.4f}')
                break

        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time

        loss_log_file.close()

        # 记录当前折的最佳结果
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch,
            'best_mse': best_mse,
            'best_ci': best_ci,
            'training_time': fold_time
        }
        fold_results.append(fold_result)
        all_fold_metrics.append([best_mse, best_ci])

        print(f'Fold {fold + 1} completed in {fold_time:.2f} seconds')
        print(f'Fold {fold + 1} Results: MSE={best_mse:.4f}, CI={best_ci:.4f}')

    # 计算交叉验证统计结果
    all_fold_metrics = np.array(all_fold_metrics)
    mean_mse = np.mean(all_fold_metrics[:, 0])
    std_mse = np.std(all_fold_metrics[:, 0])
    mean_ci = np.mean(all_fold_metrics[:, 1])
    std_ci = np.std(all_fold_metrics[:, 1])

    end_time = time.time()
    total_time = end_time - start_time

    # 保存交叉验证结果
    cv_results = {
        'dataset': dataset,
        'model': model_st,
        'n_folds': N_SPLITS,
        'fold_results': fold_results,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mean_ci': mean_ci,
        'std_ci': std_ci,
        'total_training_time': total_time
    }

    # 保存详细结果到文件
    cv_results_file = f'cv_results_{model_st}_{dataset}.pkl'
    with open(cv_results_file, 'wb') as f:
        pickle.dump(cv_results, f)

    # 保存CSV格式的汇总结果
    cv_summary_file = f'cv_summary_{model_st}_{dataset}.csv'
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(cv_summary_file, index=False)

    # 打印最终结果
    print(f'\n{"=" * 60}')
    print(f'5-FOLD CROSS-VALIDATION RESULTS FOR {dataset.upper()}')
    print(f'{"=" * 60}')
    print(f'Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}')
    print(f'Mean CI:  {mean_ci:.4f} ± {std_ci:.4f}')
    print(f'Total training time: {total_time:.2f} seconds')
    print(f'\nDetailed fold results:')
    for i, result in enumerate(fold_results):
        print(f'Fold {i + 1}: MSE={result["best_mse"]:.4f}, CI={result["best_ci"]:.4f}, '
              f'Best Epoch={result["best_epoch"]}, Time={result["training_time"]:.2f}s')
    print(f'\nResults saved to: {cv_results_file} and {cv_summary_file}')
    print(f'{"=" * 60}')
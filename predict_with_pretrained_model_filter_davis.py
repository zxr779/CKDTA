from matplotlib import pyplot as plt
import pandas as pd
import pickle
from utils import *
from create_dataset_filter import create_dataset
import time
from models.CKDTA import CKDTA
import numpy as np
from sklearn.model_selection import KFold

datasets = ['filter davis']
modelings = [CKDTA]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 32
N_SPLITS = 5  # 五折交叉验证

result = []
all_cv_results = []

for dataset in datasets:
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
    print('Complete dataset loading')

    end_time = time.time()
    all_time = end_time - start_time
    print('The data preparation took a total of ', all_time, ' seconds')

    # 设置K折交叉验证
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 预测每个modeling的结果
    for modeling in modelings:
        model_st = modeling.__name__
        print(f'\nPredicting for {dataset} using {model_st} with 5-fold cross-validation')

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

        # 存储每折的预测结果
        fold_predictions = []
        fold_metrics = []

        # 对每一折进行预测
        for fold, (train_indices, val_indices) in enumerate(kfold.split(full_data)):
            print(f'\n{"=" * 30}')
            print(f'FOLD {fold + 1}/{N_SPLITS} PREDICTION')
            print(f'{"=" * 30}')

            # 创建当前折的验证数据
            val_data_fold = [full_data[i] for i in val_indices]
            print(f'Validation samples: {len(val_data_fold)}')

            # 创建数据加载器
            val_loader = torch.utils.data.DataLoader(val_data_fold,
                                                     batch_size=TEST_BATCH_SIZE,
                                                     shuffle=False,
                                                     collate_fn=collate)

            # 加载对应折的模型
            model_file_name = f'model_{model_st}_{dataset}_fold_{fold + 1}.pth'

            if os.path.isfile(model_file_name):
                print(f'Loading model: {model_file_name}')
                model = torch.load(model_file_name, map_location=device)

                # 进行预测
                G, P = predicting(model, device, val_loader)

                # 计算指标
                fold_rmse = rmse(G, P)
                fold_mse = mse(G, P)
                fold_pearson = pearson(G, P)
                fold_spearman = spearman(G, P)
                fold_ci = ci(G, P)
                fold_rm2 = rm2(G, P)

                fold_result = {
                    'fold': fold + 1,
                    'rmse': fold_rmse,
                    'mse': fold_mse,
                    'pearson': fold_pearson,
                    'spearman': fold_spearman,
                    'ci': fold_ci,
                    'rm2': fold_rm2,
                    'true_values': G,
                    'predicted_values': P
                }

                fold_metrics.append([fold_rmse, fold_mse, fold_pearson, fold_spearman, fold_ci, fold_rm2])
                fold_predictions.append(fold_result)

                print(f'Fold {fold + 1} Results:')
                print(f'RMSE: {fold_rmse:.4f}')
                print(f'MSE: {fold_mse:.4f}')
                print(f'Pearson: {fold_pearson:.4f}')
                print(f'Spearman: {fold_spearman:.4f}')
                print(f'CI: {fold_ci:.4f}')
                print(f'RM2: {fold_rm2:.4f}')

                # 保存当前折的详细预测结果
                fold_detail_file = f'prediction_detail_{model_st}_{dataset}_fold_{fold + 1}.csv'
                fold_df = pd.DataFrame({
                    'true_values': G,
                    'predicted_values': P,
                    'absolute_error': np.abs(G - P),
                    'squared_error': (G - P) ** 2
                })
                fold_df.to_csv(fold_detail_file, index=False)
                print(f'Detailed predictions saved to: {fold_detail_file}')

            else:
                print(f'Model file not found: {model_file_name}')
                print('Please make sure you have trained the model first!')
                continue

        # 如果有有效的预测结果，计算整体统计
        if fold_metrics:
            fold_metrics = np.array(fold_metrics)

            # 计算平均值和标准差
            mean_rmse = np.mean(fold_metrics[:, 0])
            std_rmse = np.std(fold_metrics[:, 0])
            mean_mse = np.mean(fold_metrics[:, 1])
            std_mse = np.std(fold_metrics[:, 1])
            mean_pearson = np.mean(fold_metrics[:, 2])
            std_pearson = np.std(fold_metrics[:, 2])
            mean_spearman = np.mean(fold_metrics[:, 3])
            std_spearman = np.std(fold_metrics[:, 3])
            mean_ci = np.mean(fold_metrics[:, 4])
            std_ci = np.std(fold_metrics[:, 4])
            mean_rm2 = np.mean(fold_metrics[:, 5])
            std_rm2 = np.std(fold_metrics[:, 5])

            # 保存交叉验证预测结果
            cv_prediction_results = {
                'dataset': dataset,
                'model': model_st,
                'n_folds': N_SPLITS,
                'fold_predictions': fold_predictions,
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'mean_mse': mean_mse,
                'std_mse': std_mse,
                'mean_pearson': mean_pearson,
                'std_pearson': std_pearson,
                'mean_spearman': mean_spearman,
                'std_spearman': std_spearman,
                'mean_ci': mean_ci,
                'std_ci': std_ci,
                'mean_rm2': mean_rm2,
                'std_rm2': std_rm2
            }

            # 保存详细结果到文件
            cv_prediction_file = f'cv_prediction_results_{model_st}_{dataset}.pkl'
            with open(cv_prediction_file, 'wb') as f:
                pickle.dump(cv_prediction_results, f)

            # 保存CSV格式的汇总结果
            cv_prediction_summary_file = f'cv_prediction_summary_{model_st}_{dataset}.csv'
            fold_summary_data = []
            for i, fold_result in enumerate(fold_predictions):
                fold_summary_data.append({
                    'fold': fold_result['fold'],
                    'rmse': fold_result['rmse'],
                    'mse': fold_result['mse'],
                    'pearson': fold_result['pearson'],
                    'spearman': fold_result['spearman'],
                    'ci': fold_result['ci'],
                    'rm2': fold_result['rm2']
                })

            fold_summary_df = pd.DataFrame(fold_summary_data)
            fold_summary_df.to_csv(cv_prediction_summary_file, index=False)

            # 打印最终结果
            print(f'\n{"=" * 60}')
            print(f'5-FOLD CROSS-VALIDATION PREDICTION RESULTS FOR {dataset.upper()}')
            print(f'{"=" * 60}')
            print(f'Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}')
            print(f'Mean MSE:  {mean_mse:.4f} ± {std_mse:.4f}')
            print(f'Mean Pearson:  {mean_pearson:.4f} ± {std_pearson:.4f}')
            print(f'Mean Spearman: {mean_spearman:.4f} ± {std_spearman:.4f}')
            print(f'Mean CI:   {mean_ci:.4f} ± {std_ci:.4f}')
            print(f'Mean RM2:  {mean_rm2:.4f} ± {std_rm2:.4f}')
            print(f'\nDetailed fold results:')
            for i, fold_result in enumerate(fold_predictions):
                print(f'Fold {i + 1}: RMSE={fold_result["rmse"]:.4f}, MSE={fold_result["mse"]:.4f}, '
                      f'Pearson={fold_result["pearson"]:.4f}, Spearman={fold_result["spearman"]:.4f}, '
                      f'CI={fold_result["ci"]:.4f}, RM2={fold_result["rm2"]:.4f}')
            print(f'\nResults saved to: {cv_prediction_file} and {cv_prediction_summary_file}')
            print(f'{"=" * 60}')

            # 合并所有折的真实值和预测值用于整体分析
            all_true_values = np.concatenate([fold_result['true_values'] for fold_result in fold_predictions])
            all_predicted_values = np.concatenate([fold_result['predicted_values'] for fold_result in fold_predictions])

            # 计算整体指标（基于所有预测）
            overall_rmse = rmse(all_true_values, all_predicted_values)
            overall_mse = mse(all_true_values, all_predicted_values)
            overall_pearson = pearson(all_true_values, all_predicted_values)
            overall_spearman = spearman(all_true_values, all_predicted_values)
            overall_ci = ci(all_true_values, all_predicted_values)
            overall_rm2 = rm2(all_true_values, all_predicted_values)

            print(f'\nOVERALL PERFORMANCE (All Predictions Combined):')
            print(f'Overall RMSE: {overall_rmse:.4f}')
            print(f'Overall MSE:  {overall_mse:.4f}')
            print(f'Overall Pearson:  {overall_pearson:.4f}')
            print(f'Overall Spearman: {overall_spearman:.4f}')
            print(f'Overall CI:   {overall_ci:.4f}')
            print(f'Overall RM2:  {overall_rm2:.4f}')

            # 保存整体预测结果
            overall_predictions_file = f'overall_predictions_{model_st}_{dataset}.csv'
            overall_df = pd.DataFrame({
                'true_values': all_true_values,
                'predicted_values': all_predicted_values,
                'absolute_error': np.abs(all_true_values - all_predicted_values),
                'squared_error': (all_true_values - all_predicted_values) ** 2
            })
            overall_df.to_csv(overall_predictions_file, index=False)
            print(f'Overall predictions saved to: {overall_predictions_file}')

            # 添加到结果列表（用于兼容原始格式）
            ret = [dataset, model_st, round(overall_rmse, 3), round(overall_mse, 3),
                   round(overall_pearson, 3), round(overall_spearman, 3),
                   round(overall_ci, 3), round(overall_rm2, 3)]
            result.append(ret)
            all_cv_results.append(cv_prediction_results)

print('\n' + '=' * 80)
print('SUMMARY OF ALL RESULTS')
print('=' * 80)
print('dataset,model,rmse,mse,pearson,spearman,ci,rm2')
for res in result:
    print(','.join(map(str, res)))
print('=' * 80)
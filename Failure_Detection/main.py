from microservice_transformer.mit import MicroserviceTransformer
import numpy as np
import torch
import argparse
import pandas as pd
import os
from torch.utils.data import DataLoader
import math
import torch.optim as optim
import torch.nn as nn
import time
from earlyStopping import EarlyStopping
from dataclasses import dataclass

feature = ['CpuUsageRate(%)', 'MemoryUsageRate(%)', 'NodeCpuUsageRate(%)', 'Offset', 'Duration']
feature_with_label = ['CpuUsageRate(%)', 'MemoryUsageRate(%)', 'NodeCpuUsageRate(%)', 'Offset', 'Duration', 'True_Label']


"""
OB_data_len: set the number of normal data. For example, the number of normal sample data is 1390
OB_test_len: set the number of test data including normal and abnormal data.
Besides, Please set TT_data_len and TT_test_len.
"""
OB_data_len = 1390
val_ratio = 0.2
OB_test_len = 540

def detect_nan(df):
        for column in df.columns:
                nan_indices = df[column].index[df[column].isna()].tolist()
                if nan_indices:
                        print(f"column '{column}' has the Nan: {nan_indices}")
        return 0

def normalization(df):
        range_df = df.max() - df.min()
        df_normalized = df.copy()
        for column in df.columns:
                if range_df[column] == 0:
                        df_normalized[column] = 0.1
                else:
                        df_normalized[column] = (df[column] - df[column].min()) / range_df[column]
        return df_normalized

def Standardization(df):
        statistics = df.agg(['mean', 'std']).T.values.tolist()
        mean = df.mean()
        std = df.std()
        standardized_df = pd.DataFrame(columns=df.columns)
        for column in df.columns:
                if std[column] == 0:
                        print("if the standard deviation is 0, set that column to 1.")
                        standardized_df[column] = 1
                else:
                        standardized_df[column] = (df[column] - mean[column]) / std[column]
        return standardized_df, statistics


def Standardization_test(df, list_infor):
        means = [info[0] for info in list_infor]
        stds = [info[1] for info in list_infor]
        standardized_df = pd.DataFrame(columns=df.columns)
        for i in range(len(df.columns)):
                if stds[i] != 0:
                        standardized_df[df.columns[i]] = (df[df.columns[i]] - means[i]) / stds[i]
                else:
                        print("In the test set, if the standard deviation is 0, set that column to 1.")
                        standardized_df[df.columns[i]] = 1
        return standardized_df


def normalization_test_data(df):
        df_scaled = df.copy()
        normal_data = df[df['True_Label'] == 0].drop(columns=['True_Label'])
        normal_data = Handling_negative_date(normal_data)
        normal_data_scaled = normalization(normal_data)
        df_scaled.loc[df['True_Label'] == 0, normal_data_scaled.columns] = normal_data_scaled
        df_scaled = df_scaled.drop(columns=['True_Label'])
        return df_scaled


def sliding_window(data, window_size, sliding_step):
        n_samples = (data.shape[1] - window_size)//sliding_step + 1
        n_features = data.shape[0]
        n_micro = data.shape[2]
        result = np.zeros((n_samples, n_features, window_size, n_micro))
        for i in range(n_samples):
                start_index = i * sliding_step
                result[i] = data[:, start_index:start_index + window_size, :]
        return result

def Handling_negative_date(df):
        for column in df.columns:
                min_value = df[column].min()
                if min_value < 0:
                        df[column] = df[column] - min_value
        return df


def read_train_data(normal_path1, normal_path2, config):
        normalization_flag = config.normalization
        standardization_flag = config.standardization
        csv_name1 = os.listdir(normal_path1)
        csv_name2 = os.listdir(normal_path2)
        global microservice_num
        microservice_num = len(csv_name1)
        global feature_num
        feature_num = len(feature)

        dataset_name = config.dataset
        if dataset_name == 'OB':
                sorted_csv_name1 = sorted(csv_name1)
                sorted_csv_name2 = sorted(csv_name2)
                val_len = math.floor(OB_data_len * val_ratio)
                data_len = OB_data_len - val_len
                normalization_flag = True
                standardization_flag = False
        if dataset_name == 'TT':
                sorted_csv_name1 = sorted(csv_name1, key=lambda x: x.split('ts-')[1].split('-service')[0])
                sorted_csv_name2 = sorted(csv_name2, key=lambda x: x.split('ts-')[1].split('-service')[0])
                val_len = math.floor(TT_data_len * val_ratio)
                data_len = TT_data_len - val_len
                standardization_flag = True
                normalization_flag = False

        train_data_array = np.zeros((feature_num, data_len, microservice_num))
        val_data_array = np.zeros((feature_num, val_len, microservice_num))
        print("Shape of Normal data for training: ", train_data_array.shape)
        print("Shape of Normal data for validation: ", val_data_array.shape)

        statistics_dict1 = {}
        statistics_dict2 = {}
        print("train standardization_flag: ", standardization_flag)
        print("train normalization_flag: ", normalization_flag)
        for i, file_name in enumerate(sorted_csv_name1):
                path1 = normal_path1 + file_name
                df = pd.read_csv(path1, usecols=feature)
                detect_nan(df)
                df = Handling_negative_date(df)
                if normalization_flag and not standardization_flag:
                        df = normalization(df)
                elif standardization_flag and not normalization_flag:
                        df, statistics = Standardization(df)
                        statistics_dict1[i] = statistics
                else:
                        df = df

                each_train_len = df.shape[0] - math.floor(val_len/2)
                df_train = df.iloc[:each_train_len]
                df_val = df.iloc[each_train_len:]
                normal_train_data1_len = df_train.shape[0]
                normal_val_data1_len = df_val.shape[0]
                train_data_array[:, :normal_train_data1_len, i] = df_train.values.T
                val_data_array[:, :normal_val_data1_len, i] = df_val.values.T


        for i, file_name in enumerate(sorted_csv_name2):
                path2 = normal_path2 + file_name
                df = pd.read_csv(path2, usecols=feature)
                detect_nan(df)
                df = Handling_negative_date(df)
                if normalization_flag and not standardization_flag:
                        df = normalization(df)

                elif standardization_flag and not normalization_flag:
                        df, statistics = Standardization(df)
                        statistics_dict2[i] = statistics
                else:
                        df = df

                each_train_len = df.shape[0] - math.floor(val_len / 2)
                df_train = df.iloc[:each_train_len]
                df_val = df.iloc[each_train_len:]
                normal_train_data2_len = df_train.shape[0]
                normal_val_data2_len = df_val.shape[0]
                train_data_array[:, -normal_train_data2_len:, i] = df_train.values.T
                val_data_array[:, -normal_val_data2_len:, i] = df_val.values.T

        global train_mean_std_dict
        train_mean_std_dict = {}
        for key in statistics_dict1.keys():
                train_mean_std_dict[key] = [
                        [(a + b) / 2 for a, b in zip(pair1, pair2)]
                        for pair1, pair2 in zip(statistics_dict1[key], statistics_dict2[key])
                ]

        # train_data_array val_data_array, sliding window
        window_size = config.win_size
        sliding_step = config.sliding_step
        train_dataset = sliding_window(train_data_array, window_size, sliding_step) #(batchsize, feature, window_size, microservice)
        val_dataset = sliding_window(val_data_array, window_size, sliding_step)


        # to tensor
        train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
        val_dataset = torch.tensor(val_dataset, dtype=torch.float32)
        print()
        print("Training windows shape by sliding windows:", train_dataset.shape)
        print("Validation windows shape by sliding windows:", val_dataset.shape)


        batch_size = config.batch_size
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0)
        return train_loader, val_loader

def sliding_window_label(test_label_array, window_size, sliding_step):
    new_length = (test_label_array.shape[0] - window_size) // sliding_step + 1
    new_label_array = np.zeros((new_length, 1), dtype=int)
    for i in range(new_length):
        start_index = i * sliding_step
        end_index = start_index + window_size
        window = test_label_array[start_index:end_index]
        if np.all(window == 0):
            new_label_array[i] = 0
        else:
            new_label_array[i] = 1
    return new_label_array

def read_test_data(abnormal_path1, true_labels_path1, abnormal_path2, true_labels_path2, config):
        #read test labels
        standardization_flag = config.standardization
        normalization_flag = config.normalization
        window_size = config.win_size
        sliding_step = config.sliding_step
        df_labels1 = pd.read_csv(true_labels_path1, usecols=['True_Label'])
        df_labels2 = pd.read_csv(true_labels_path2, usecols=['True_Label'])
        test_labels = pd.concat([df_labels1, df_labels2], ignore_index=True).values
        print("test_labels: ", test_labels.shape)

        csv_name1 = os.listdir(abnormal_path1)
        csv_name2 = os.listdir(abnormal_path2)

        global microservice_num
        microservice_num = len(csv_name1)
        global feature_num
        feature_num = len(feature)

        dataset_name = config.dataset
        global sorted_csv_name1
        if dataset_name == 'OB':
                sorted_csv_name1 = sorted(csv_name1)
                sorted_csv_name2 = sorted(csv_name2)
                test_len = OB_test_len
                normalization_flag = True
                standardization_flag = False

        if dataset_name == 'TT':
                sorted_csv_name1 = sorted(csv_name1, key=lambda x: x.split('ts-')[1].split('-service')[0])
                sorted_csv_name2 = sorted(csv_name2, key=lambda x: x.split('ts-')[1].split('-service')[0])
                test_len = TT_test_len
                standardization_flag = True
                normalization_flag = False

        print("test standardization_flag: ", standardization_flag)
        print("test normalization_flag: ", normalization_flag)
        test_data_array = np.zeros((feature_num, test_len, microservice_num))


        for i, file_name in enumerate(sorted_csv_name1):
                path1 = abnormal_path1 + file_name
                df_file1 = pd.read_csv(path1, usecols=feature_with_label)

                detect_nan(df_file1)
                if normalization_flag and not standardization_flag:
                        df_file1 = normalization_test_data(df_file1)

                elif standardization_flag and not normalization_flag:
                        df_file1 = df_file1.drop(columns=['True_Label'])
                        df_file1 = Handling_negative_date(df_file1)
                        train_infor_in_each_file = train_mean_std_dict[i]
                        df_file1 = Standardization_test(df_file1, train_infor_in_each_file)
                else:
                        df_file1 = df_file1.drop(columns=['True_Label'])
                        df_file1 = Handling_negative_date(df_file1)

                abnormal_test_data1_len = df_file1.shape[0]
                test_data_array[:, :abnormal_test_data1_len, i] = df_file1.values.T


        for i, file_name in enumerate(sorted_csv_name2):
                path2 = abnormal_path2 + file_name
                df_file2 = pd.read_csv(path2, usecols=feature_with_label)
                detect_nan(df_file2)
                if normalization_flag and not standardization_flag:
                        df_file2 = normalization_test_data(df_file2)
                elif standardization_flag and not normalization_flag:
                        df_file2 = df_file2.drop(columns=['True_Label'])
                        df_file2 = Handling_negative_date(df_file2)
                        train_infor_in_each_file = train_mean_std_dict[i]
                        df_file2 = Standardization_test(df_file2, train_infor_in_each_file)
                else:
                        df_file2 = df_file2.drop(columns=['True_Label'])
                        df_file2 = Handling_negative_date(df_file2)

                abnormal_test_data2_len = df_file2.shape[0]
                test_data_array[:, -abnormal_test_data2_len:, i] = df_file2.values.T
        test_dataset = sliding_window(test_data_array, window_size, sliding_step)

        # to tensor
        test_dataset = torch.tensor(test_dataset, dtype=torch.float32)

        print("Testing windows shape by sliding windows:", test_dataset.shape)

        batch_size = config.batch_size
        test_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

        return test_loader, test_labels


def compute_loss(batch_max, outputs):
        squared_diff = (batch_max - outputs) ** 2
        batch_loss = squared_diff.sum(dim=1)
        get_loss = batch_loss.mean()
        return get_loss


def train_and_test(train_loader, val_loader, test_loader, test_labels, config):
        #intial model
        test_epoch = config.test_epoch
        window_length = config.win_size
        embedding_dim = microservice_num * feature_num
        step_size = config.lr_step_size
        lr = config.lr
        gamma = config.gamma

        mit_model = MicroserviceTransformer(
                window_length=window_length,
                feature_num=feature_num,
                microservice_num=microservice_num,
                embedding_dim=embedding_dim,
                num_layers=4,
                num_heads=feature_num,
                qkv_bias=True,
                mlp_ratio=4.0,
                use_revised_ffn=False,
                dropout_rate=0.3,
                attn_dropout_rate=0.2,
                cls_head=False)

        mse_loss = nn.MSELoss()
        es = EarlyStopping(mode='min', min_delta=0.0, patience=config.patience)
        optimizer = optim.Adam(mit_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # start training
        print("======================Start Train and Validation======================")
        num_epochs = config.num_epochs
        train_start_time = time.time()
        for epoch in range(num_epochs):
                mit_model.train()
                train_loss_sum, train_num = 0.0, 0
                val_loss_sum, val_num = 0.0, 0
                for train_batch in train_loader:
                        optimizer.zero_grad()
                        train_outputs = mit_model(train_batch)  #torch.Size([bitchsize, embedding_dim])
                        
                        outputs_nan = torch.isnan(train_outputs).any()
                        if (outputs_nan == True):
                                print("outputs_nan: ", outputs_nan)

                        normal_batch_flatten = train_batch.reshape(train_batch.size(0), train_batch.size(2), -1)
                        normal_batch_max, indices = torch.max(normal_batch_flatten, dim=1)
                        train_loss = mse_loss(normal_batch_max, train_outputs)

                        train_loss.backward()
                        optimizer.step()


                        train_loss_sum += train_loss.item() * train_batch.shape[0]
                        train_num += train_batch.shape[0]
                avg_train_loss = train_loss_sum / train_num
                scheduler.step()
                train_end_time = time.time()
                restruction_train_time = train_end_time - train_start_time
                restruction_train_time = restruction_train_time / 60

                # Validate the model
                mit_model.eval()
                val_loss_list = []
                with torch.no_grad():
                        for val_batch in val_loader:
                                val_outputs = mit_model(val_batch)
                                val_batch_flatten = val_batch.reshape(val_batch.size(0), val_batch.size(2), -1)
                                val_batch_max, indices = torch.max(val_batch_flatten, dim=1)
                                val_loss = mse_loss(val_batch_max, val_outputs)
                                val_loss_sum += val_loss.item() * val_batch.shape[0]
                                val_num += val_batch.shape[0]
                                val_loss_list.append(val_loss.item())

                avg_val_loss = val_loss_sum / val_num
                max_val_loss = max(val_loss_list)
                print(
                        'Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | Train time/min: {:.3f}'. \
                                format(epoch + 1, optimizer.param_groups[0]['lr'], avg_train_loss, avg_val_loss,
                                       restruction_train_time))

                if es.step(val_loss):
                        print('Early stopping.')

                if (epoch + 1) % test_epoch == 0:
                        print("====================Start Testing========================================")
                        test(mit_model, test_loader, test_labels, epoch + 1, config, max_val_loss)
        return


def get_pre_label(test_outputs, test_batch_max, max_val_loss, dataset, top_k):
        pre_label_num = test_outputs.shape[0]
        pre_labels = np.zeros((pre_label_num, 1))-1
        anomaly_location = np.zeros((pre_label_num, top_k))-1

        squared_diff = (test_batch_max - test_outputs) ** 2
        test_batch_loss = squared_diff.sum(dim=1)/squared_diff.size(1)
        print("max_val_loss: ", max_val_loss)
        index = 0
        if dataset == 'OB':
                u = 11
                val_loss = max_val_loss + max_val_loss * u

        if dataset == 'TT':
                u1 = 100
                val_loss = max_val_loss + max_val_loss * u1

        print("threshold: ", val_loss)
        i = 0
        for score in test_batch_loss:
                squared_diff_each_batch = squared_diff[i]
                if score.item() > val_loss:
                        topk_values, topk_indices = torch.topk(squared_diff_each_batch, k=top_k)
                        topk_indices = topk_indices.numpy()
                        anomaly_location[index] = topk_indices
                        pre_labels[index] = 1

                else:
                        pre_labels[index] = 0
                index = index + 1
                i = i + 1

        return pre_labels, anomaly_location

@dataclass
class EvalResults:
    TP: int
    FP: int
    TN: int
    FN: int
    precision: float
    FDR: float
    FPR: float
    TPR: float

def evalation(new_pre_labels, true_test_labels) -> EvalResults:
        new_pre_labels = new_pre_labels.flatten()
        true_test_labels = true_test_labels.flatten()
        print("new_pre_labels： ")
        print(new_pre_labels.shape)
        print("true_test_labels")
        print(true_test_labels.shape)
        TP = np.sum((new_pre_labels == 1) & (true_test_labels == 1))
        FP = np.sum((new_pre_labels == 1) & (true_test_labels == 0))
        TN = np.sum((new_pre_labels == 0) & (true_test_labels == 0))
        FN = np.sum((new_pre_labels == 0) & (true_test_labels == 1))

        precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
        FDR = FP / (TP + FP) * 100 if (TP + FP) > 0 else 100
        FPR = FP / (FP + TN) * 100 if (FP + TN) > 0 else 100
        TPR = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
        return EvalResults(TP, FP, TN, FN, precision, FDR, FPR, TPR)



def interpretable(anomaly_location_list, dataset):
        flattened_list = [anomaly_location_list[i][j] for i in range(len(anomaly_location_list)) for j in range(anomaly_location_list[i].shape[0])]
        file_name = 'interpretable_anomaly_' + dataset + '.txt'
        for i in range(len(flattened_list)):
                data_t = flattened_list[i]
                if np.all(data_t != -1):
                        with open(file_name, 'a') as f:
                                f.write("In the entire test set, the {}th timestamp was classified as an anomaly. The top three microservices and their corresponding metrics are detailed below:\n".format(i))
                                for idx in data_t:
                                        feature_index = int(idx) // len(sorted_csv_name1)
                                        microservice_index = int(idx) % len(sorted_csv_name1)
                                        feat = feature[feature_index]
                                        microservice = sorted_csv_name1[microservice_index]
                                        f.write(f"Index {idx} -> Feature: {feat}, Microservice: {microservice}\n")
                                f.write('\n')
        print("The anomaly explanation results have been saved.")
        return 0

@torch.no_grad()
def test(mit_model, test_loader, test_labels, train_epochs, config, max_val_loss):
        top_k = config.top_k
        test_start_time = time.time()
        mit_model.eval()
        pre_labels_list = []
        anomaly_location_list = []
        win_size = config.win_size
        sliding_step = config.sliding_step
        dataset = config.dataset
        test_labels_num = test_labels.shape[0]
        flag = 0
        for test_batch in test_loader: #(batchsize, feature_num, win_size, microservice)
                print("flag: ", flag)
                test_outputs = mit_model(test_batch) #(batchsize, feature_num * microservice)
                test_batch_flatten = test_batch.reshape(test_batch.size(0), test_batch.size(2), -1)
                test_batch_max, indices = torch.max(test_batch_flatten, dim=1) #(batchsize, feature_num * microservice)

                pre_labels, anomaly_location = get_pre_label(test_outputs, test_batch_max, max_val_loss, dataset, top_k)
                pre_labels_list.append(pre_labels)
                anomaly_location_list.append(anomaly_location)
                flag = flag + 1


        pre_labels_list = np.vstack(pre_labels_list)
        count_of_ones = np.sum(pre_labels_list)
        print("The length of pre_labels_list: ", len(pre_labels_list))
        print("The number of anomaly(1) in pre_labels_list: ", count_of_ones)
        print("The number of normal (0) in pre_labels_list: ", len(pre_labels_list)-count_of_ones)

        df_pre_labels_list = pd.DataFrame(pre_labels_list, columns=['Pre_Label'])

        file_name = 'pre_label_' + dataset + '.csv'
        df_pre_labels_list.to_csv(file_name, index=False)
        print("The prediction results have been saved.")


        true_label_win = sliding_window_label(test_labels, win_size, sliding_step)
        df_true_label_win = pd.DataFrame(true_label_win, columns=['New_True_Label'])

        true_file_name = 'true_label_' + dataset + '.csv'
        df_true_label_win.to_csv(true_file_name, index=False)
        print("The true label data has been saved.")


        print("The length of the true labels after applying the sliding window mechanism: ", len(true_label_win))
        count_of_ones = np.sum(true_label_win)
        print("The length of true_label_win: ", len(true_label_win))
        print("The number of anomaly(1) in true_label_win: ", count_of_ones)
        print("The number of normal(0) in true_label_win: ", len(true_label_win) - count_of_ones)


        eval_result = evalation(pre_labels_list, true_label_win)
        test_TP = eval_result.TP
        test_FP = eval_result.FP
        test_TN = eval_result.TN
        test_FN = eval_result.FN
        test_precision = eval_result.precision
        test_FDR = eval_result.FDR
        test_FPR = eval_result.FPR
        test_TPR = eval_result.TPR

        test_end_time = time.time()
        detection_test_time = test_end_time - test_start_time
        detection_test_time = detection_test_time  #second

        print(
                "Test Epoch {:03d} | Test TP {} | Test FP {} | Test TN {}| Test FN {}| Test Pre {:.2f}% | Test FDR {:.2f}% | Test FPR {:.2f}% | Test TPR {:.2f}% |Test time/sec {:.3f}".format(
                        train_epochs,  test_TP, test_FP, test_TN, test_FN, test_precision, test_FDR, test_FPR, test_TPR,
                        detection_test_time))

        print("===================================interpretable anomaly results========================================")
        interpretable(anomaly_location_list, dataset)

        return 0


def main(config):
        """
        normal_data_: contains normal data for training
        abnormal_data_: contains normal and abnormal data for testing
        true_label_: contains true labels of test data
        """
        if config.dataset == 'OB':
                Dataset_path = "./OB_Sample_Data"
                normal_data_path1 = Dataset_path + "/normal_data_01/"
                normal_data_path2 = Dataset_path + "/normal_data_02/"
                abnormal_data_path1 = Dataset_path + "/abnormal_data_01/"
                abnormal_data_path2 = Dataset_path + "/abnormal_data_01/"
                true_labels_path1 = Dataset_path + '/true_label_01.csv'
                true_labels_path2 = Dataset_path + '/true_label_02.csv'

        elif config.dataset == 'TT':
                Dataset_path = "./TT_Sample_Data"
                normal_data_path1 = Dataset_path + "/normal_data_01/"
                normal_data_path2 = Dataset_path + "/normal_data_02/"
                abnormal_data_path1 = Dataset_path + "/abnormal_data_01/"
                abnormal_data_path2 = Dataset_path + "/abnormal_data_02/"
                true_labels_path1 = Dataset_path + '/true_label_01.csv'
                true_labels_path2 = Dataset_path + '/true_label_02.csv'
        else:
                print("Please input: --dataset OB/TT")


        print("======================Data Preparation======================")
        """
        Note: We aggregate the datasets from the two paths for model training and testing.
        """
        train_loader, val_loader = read_train_data(normal_data_path1, normal_data_path2, config)    # Use normal data in train stage.
        test_loader, test_labels = read_test_data(abnormal_data_path1, true_labels_path1, abnormal_data_path2, true_labels_path2, config)
        train_and_test(train_loader, val_loader, test_loader, test_labels, config)
        return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--win_size', type=int, default=3)
    parser.add_argument('--sliding_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='OB')
    parser.add_argument('--test_epoch', type=int, default=100)
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--normalization', action='store_true', help='Enable the normalization (default: False)')
    parser.add_argument('--standardization', action='store_true', help='Enable the standardization (default: False)')

    config = parser.parse_args()
    main(config)

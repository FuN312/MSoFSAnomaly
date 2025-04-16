import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

"""
For each microservice, visualize the relationship between different features and faults, using the frontend microservice as an example.
"""

Metrics_and_labels = ["CpuUsageRate(%)", "MemoryUsageRate(%)", "SyscallRead", "SyscallWrite", "NetworkReceiveBytes", "NetworkTransmitBytes", "PodClientLatencyP90(s)",
            "PodServerLatencyP90(s)", "PodWorkload(Ops)", "NodeCpuUsageRate(%)", "NodeMemoryUsageRate(%)", "NodeNetworkReceiveBytes", "label"]


front_fault_labels = ["a", "b", "c", "d"]  # a:cpu_contention b: error_return c: cpu_consumed d: code_exception
data_num = 300

def find_continuous_ones_indices(label_df):
    indices = np.where(label_df == 1)[0]
    splits = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    splits = [split for split in splits if len(split) > 1]
    return [split.tolist() for split in splits]

def read_csv_to_draw(directory):
    files = [f for f in os.listdir(directory) if f.startswith('OB_') and f.endswith('.csv')]
    for file in files:
        if(file == "OB_frontend-all-metric.csv"):
            print("csv file: ", file)
            df = pd.read_csv(os.path.join(directory, file))
            df = df[Metrics_and_labels]
            df = df.head(data_num)

            label_df = df['label']
            df.drop(columns=['label'], inplace=True)
            df_normalized = (df - df.min()) / (df.max() - df.min())
            Metrics = Metrics_and_labels[:-1]

            for i in range(len(Metrics)):
                metrics_name = Metrics[i]

                data_df = df[metrics_name]
                data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())

                indices_lists = find_continuous_ones_indices(label_df)
                for idx, indices in enumerate(indices_lists, 1):
                    print(f"Window {idx}: {indices}")
                # Plotting
                plt.figure()
                plt.plot(data_df.index, data_df)

                x = np.linspace(1, data_num, data_num)


                for rect_x, label in zip(indices_lists, front_fault_labels):
                    start = min(rect_x)
                    end = max(rect_x)
                    plt.fill_between(x[start - 1:end], np.min(data_df), np.max(data_df), color='gray', alpha=0.3)

                    # 添加标签
                    label_x = np.mean(x[start - 1:end])
                    label_y = np.max(data_df) + 0.01
                    plt.text(label_x, label_y, label, ha='center', va='bottom', fontsize=10, color='black', weight='bold')


                # 添加标题和坐标轴标签
                plt.xlabel('Timestamp/minutes')
                plt.ylabel(metrics_name)
                #plt.legend()

                plt.show()
    return


# Read all CSV files in the current directory (change directory as needed)
directory = './'
read_csv_to_draw(directory)



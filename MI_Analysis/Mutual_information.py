from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd

"""
Calculate the MI value for each type of fault, using CPU Contention from OB system as an example.
"""


data_num = 300
Metrics_and_labels = ["CpuUsageRate(%)", "MemoryUsageRate(%)", "SyscallRead", "SyscallWrite", "NetworkReceiveBytes", "NetworkTransmitBytes", "PodClientLatencyP90(s)",
            "PodServerLatencyP90(s)", "PodWorkload(Ops)", "NodeCpuUsageRate(%)", "NodeMemoryUsageRate(%)", "NodeNetworkReceiveBytes", "label"]


cpu_contention_service_name = ["cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "paymentservice", "recommendationservice", "shippingservice"]
result = {}
for i in range(len(cpu_contention_service_name)):
    service_name = cpu_contention_service_name[i]
    file = "OB_" + service_name + "-all-metric.csv"  # Each file here contains metric features of a microservice, including labels that distinguish between normal and abnormal states.
    print("file: ", file)
    df = pd.read_csv(file)
    df = df[Metrics_and_labels]
    df = df.head(data_num)
    label_df = df['label']
    y = label_df.values
    df.drop(columns=['label'], inplace=True)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    x = df_normalized.values

    # Compute MI
    mi = mutual_info_classif(x, y)
    mutual_score = list(mi)
    mutual_score = [round(value, 4) for value in mutual_score]
    metrics_dict = dict(zip(Metrics_and_labels, mutual_score))
    # Use the sorted function and the lambda function to sort a dictionary by its values in descending order.
    sorted_metrics_dict = dict(sorted(metrics_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_metrics_dict)
    result[service_name] = sorted_metrics_dict
    print("-------------------------------------")

print(result)

Metrics = Metrics_and_labels[:-1]
avg_metric = {}

for metric in Metrics:
    total = 0
    for service in result.values():
        total += service.get(metric, 0)  #Use the get method to avoid a KeyError
    avg_metric[metric] = round(total / len(result), 4)

avg_sorted_metrics = dict(sorted(avg_metric.items(), key=lambda item: item[1], reverse=True))
print(avg_sorted_metrics)

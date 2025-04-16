import pandas as pd
import numpy as np
from os.path import dirname
import statistics
import os
import re
import json
from datetime import datetime

"""
Extract the Span Duration features from the trace data.
"""

def get_netwrok_metric(trace_file, pod_name):
    latency_list = []
    print("pod_name", pod_name)
    pod_reader = pd.read_csv(trace_file, usecols=['SpanID', 'ParentID', 'PodName', 'Duration'], index_col='PodName')
    pod_reader['PodName'] = pod_reader.index

    try:
        pod_spans = pod_reader.query(f'PodName == "{pod_name}"')[['SpanID', 'ParentID', 'PodName', 'Duration']]
    except:
        print("The podName is not found, read the predefined threshold value.")
        return 0
    if len(pod_spans['SpanID']) > 0:
        latency_list.append(pod_spans["Duration"])
    if len(latency_list) > 0:
        return np.percentile(latency_list, 90)
    else:
        return 0

def compute_trace_name(fault_inject_data):
    results = []
    for hour_list in fault_inject_data.values():
        for event in hour_list:
            inject_pod = event["inject_pod"]
            inject_time_str = event["inject_time"]

            inject_time_dt = datetime.strptime(inject_time_str, "%Y-%m-%d %H:%M:%S")
            hour_minute_str = inject_time_dt.strftime("%H:%M")
            hour, minute = hour_minute_str.split(':')
            hour = hour.zfill(2)

            inject_time1 = f"{hour}_{minute}"
            inject_time2 = f"{hour}_{int(minute) + 1:02d}"
            inject_time3 = f"{hour}_{int(minute) + 2:02d}"

            results.append({
                "inject_pod": inject_pod,
                "inject_time1": inject_time1,
                "inject_time2": inject_time2,
                "inject_time3": inject_time3
            })
    return results

metric_dir = "./metric/"
trace_dir = "./trace/"
latency_csv_path = "./Span_Duration/"

metric_csv_name = os.listdir(metric_dir)
pod_name = [item.replace('_metric.csv', '') for item in metric_csv_name]

inject_fault_json = "fault_list.json"
f = open(inject_fault_json)
fault_inject_data = json.load(f)
trace_csv_name = compute_trace_name(fault_inject_data)



for i in range(len(pod_name)):
    service_duration90 = []
    inject_time_list = []
    service_pod_name = pod_name[i]
    print("service_pod_name: ", service_pod_name)
    for j in range(len(trace_csv_name)):
        dic = trace_csv_name[j]
        values = dic.values()

        for index, value in enumerate(values):
            if index > 0:
                inject_time = value
                csv_file_name = inject_time+"_trace.csv"
                trace_csv_path = trace_dir+csv_file_name
                service_duration = get_netwrok_metric(trace_file=trace_csv_path, pod_name=service_pod_name)
                print("service | {}, in fault time | {}, latency | {}".format(service_pod_name, inject_time, service_duration))
                service_duration90.append(service_duration)
                inject_time_list.append(inject_time)

    df = pd.DataFrame()
    df['Fault_Timestamp'] = inject_time_list
    df['network_latency90'] = service_duration90

    latency_csv_name = service_pod_name + "_duration.csv"
    csv_file = latency_csv_path + latency_csv_name
    df.to_csv(csv_file, index=False)
    print(f"DataFrame with titles saved to '{csv_file}'")






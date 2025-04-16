import pandas as pd
import numpy as np
from os.path import dirname
import statistics
import os
import re
import json
from datetime import datetime

"""
Extract the Span Offset features from the trace data.
"""

def get_netwrok_metric(trace_file, pod_name):
    offset_list = []
    if "front" in pod_name: #front end dose not calculate offset latency
        return 10, 10


    pod_reader = pd.read_csv(trace_file, usecols=['TraceID', 'SpanID', 'ParentID', 'PodName', 'EndTimeUnixNano'], index_col='PodName')
    pod_reader['PodName'] = pod_reader.index

    parent_span_reader = pd.read_csv(trace_file, usecols=['TraceID', 'SpanID', 'ParentID', 'PodName', 'EndTimeUnixNano'], index_col='SpanID')
    parent_span_reader['SpanID'] = parent_span_reader.index

    try:
        pod_spans = pod_reader.query(f'PodName == "{pod_name}"')[['SpanID', 'ParentID', 'PodName', 'EndTimeUnixNano']]
    except:
        print("The podName is not found, read the predefined threshold value.")
        service = pod_name.rsplit('-', 1)[0]
        service = service.rsplit('-', 1)[0]

        current_path = os.getcwd()
        parent_path = os.path.dirname(current_path)
        path = parent_path.replace('\\', '/')
        csv_file = path + "/metric_threshold/" + service + ".csv"
        pod_reader = pd.read_csv(csv_file, usecols=['NetworkP90(ms)'])
        return float(pod_reader.iloc[0]), 0


    if len(pod_spans['SpanID']) > 0:
        for span_index in range(len(pod_spans['SpanID'])):
            # span event
            parent_id = pod_spans['ParentID'].iloc[span_index]
            pod_start_time = int(pod_spans['EndTimeUnixNano'].iloc[span_index])
            try:
                parent_pod_span = parent_span_reader.loc[[parent_id], ['PodName', 'EndTimeUnixNano']]
                if len(parent_pod_span) > 0:
                    for parent_span_index in range(len(parent_pod_span['PodName'])):
                        parent_pod_name = parent_pod_span['PodName'].iloc[parent_span_index]
                        parent_end_time = int(parent_pod_span['EndTimeUnixNano'].iloc[parent_span_index])

                    if str(parent_pod_name) != str(pod_name):
                        offset = (parent_end_time - pod_start_time) / 1000000
                        offset_list.append(offset)
            except:
                pass

    if len(offset_list) > 2:
        return np.percentile(offset_list, 90), statistics.stdev(offset_list)
    else:
        return 10, 10

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
latency_csv_path = "./Span_Offset/"

metric_csv_name = os.listdir(metric_dir)
pod_name = [item.replace('_metric.csv', '') for item in metric_csv_name]

inject_fault_json = "fault_list.json"
f = open(inject_fault_json)
fault_inject_data = json.load(f)
trace_csv_name = compute_trace_name(fault_inject_data)



for i in range(len(pod_name)):
    offset90 = []
    offset_stdev = []
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
                offset_p90, offset_p_stdev = get_netwrok_metric(trace_file=trace_csv_path, pod_name=service_pod_name)
                print("service | {}, in fault time | {}, offset | {}".format(service_pod_name, inject_time, offset_p90))
                offset90.append(offset_p90)
                offset_stdev.append(offset_p_stdev)
                inject_time_list.append(inject_time)

    df = pd.DataFrame()
    df['Fault_Timestamp'] = inject_time_list
    df['offset'] = offset90
    df['offset_stdev'] = offset_stdev
    latency_csv_name = service_pod_name + "_offset.csv"
    csv_file = latency_csv_path + latency_csv_name
    df.to_csv(csv_file, index=False)
    print(f"DataFrame with titles saved to '{csv_file}'")






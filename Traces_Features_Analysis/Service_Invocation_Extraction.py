import pandas as pd
import numpy as np
import json
from datetime import datetime

"""
From the trace data, first extract the calling relationships between microservices for each trace, 
then obtain the unique microservice call relationships. 
This will support the identification of fault-sensitive microservices.
"""

def compute_trace_name(fault_inject_data):
    results = []
    for hour_list in fault_inject_data.values():
        for event in hour_list:
            inject_pod = event["inject_pod"]
            inject_time_str = event["inject_time"]

            # Parse the inject_time string into a datetime object
            inject_time_dt = datetime.strptime(inject_time_str, "%Y-%m-%d %H:%M:%S")

            # Extract the hours and minutes and format them as "04:53"
            hour_minute_str = inject_time_dt.strftime("%H:%M")

            # Split the hours and minutes, ensuring the hour part is two digits
            hour, minute = hour_minute_str.split(':')
            hour = hour.zfill(2)
            inject_time1 = f"{hour}_{minute}"

            minute = int(minute)
            minute1 = minute + 1
            if minute1 >= 60:
                hour1 = str(int(hour) + 1).zfill(2)
                minute1 = 0
            else:
                hour1 = hour.zfill(2)
            inject_time2 = f"{hour1}_{minute1:02d}"

            minute2 = minute + 2
            if minute2 >= 60:
                hour2 = str(int(hour) + 1).zfill(2)
                minute2 = minute2 % 60
            else:
                hour2 = hour.zfill(2)
            inject_time3 = f"{hour2}_{minute2:02d}"

            results.append({
                "inject_pod": inject_pod,
                "inject_time1": inject_time1,
                "inject_time2": inject_time2,
                "inject_time3": inject_time3
            })
    return results

def read_csv_divde_trace(trace_file):
    pod_reader = pd.read_csv(trace_file, usecols=['TraceID', 'SpanID', 'ParentID', 'PodName'])

    # Initialize dictionary to store partitions by traceid
    trace_partitions = {}

    # Partition data by traceid
    for index, row in pod_reader.iterrows():
        trace_id = row['TraceID']
        if trace_id not in trace_partitions:
            trace_partitions[trace_id] = []
        trace_partitions[trace_id].append({
            'SpanID': row['SpanID'],
            'ParentID': row['ParentID'],
            'PodName': row['PodName']
        })
    return trace_partitions

def get_service_invocation(trace_data):
    call_list = []

    # Create a dictionary to quickly lookup PodName by SpanID
    spanid_to_podname = {item['SpanID']: item['PodName'] for item in trace_data}

    # Process each span data in trace_data
    for item in trace_data:
        parent_id = item['ParentID']
        pod_name = item['PodName']
        service = pod_name.rsplit('-', 1)[0]
        pod_microservice = service.rsplit('-', 1)[0]

        if parent_id in spanid_to_podname:
            parent_pod_name = spanid_to_podname[parent_id]
            service = parent_pod_name.rsplit('-', 1)[0]
            pod_parent_name = service.rsplit('-', 1)[0]

            if pod_microservice != pod_parent_name:  # Avoid self-calls
                call_list.append([pod_parent_name, pod_microservice])

    return call_list

def get_unique_call(call_unique_result_path, result_service_call):
    unique_tuples = set()

    # Iterate over each traceid's call_list in result
    for traceid, call_list in result_service_call.items():
        for tuple_item in call_list:
            # Convert each list to a tuple for immutability and add to the set
            unique_tuples.add(tuple(tuple_item))
    unique_list = list(unique_tuples)
    print("unique_list num", len(unique_list))
    print("---------------------------------------------------")

    with open(call_unique_result_path, 'w') as f:
        json.dump(unique_list, f, indent=2)



trace_dir = "./trace/"
call_csv_path = "./Call_service/"
call_unique_csv_path = "./Call_service_unique/"



inject_fault_json = "fault_list.json"
f = open(inject_fault_json)
fault_inject_data = json.load(f)
trace_csv_name = compute_trace_name(fault_inject_data)


for j in range(len(trace_csv_name)):
    dic = trace_csv_name[j]
    values = dic.values()

    for index, value in enumerate(values):
        if index > 0:
            result_service_call = {}
            inject_time = value
            csv_file_name = inject_time+"_trace.csv"
            trace_csv_path = trace_dir+csv_file_name
            print(trace_csv_path)
            trace_partitions = read_csv_divde_trace(trace_csv_path)

            for trace_id, trace_data in trace_partitions.items():
                call_list = get_service_invocation(trace_data)
                if(len(call_list)!=0):
                    result_service_call[trace_id] = call_list
            print("file_name {} | trace_num {} | trace_num_with_span {}".format(csv_file_name, len(trace_partitions),
                                                                                len(result_service_call)))
            call_result_name = inject_time+"_trace.json"
            call_result_path = call_csv_path + call_result_name
            with open(call_result_path, 'w') as f:
                json.dump(result_service_call, f, indent=2)

            call_unique_result_path = call_unique_csv_path + call_result_name
            get_unique_call(call_unique_result_path, result_service_call)






import pandas as pd
import os
import json

"""
Extract the calling relationship between parent spans and child spans from the trace data.
"""

def get_order_time(value):
    parent_start_time = value['parent_start_time']
    parent_end_time = value['parent_end_time']
    children = value['children']

    a = {}

    a['parent_start_time'] = parent_start_time
    a['parent_end_time'] = parent_end_time

    children_num = len(children)
    for i in range(children_num):
        each_child = children[i]
        start_str = "child_start_time" + str(i + 1)
        end_str = "child_end_time" + str(i + 1)

        a[start_str] = each_child['start_time']
        a[end_str] = each_child['end_time']

    sorted_items = sorted(a.items(), key=lambda x: x[1])
    return sorted_items

def get_Parent_and_Son(trace_csv_path, json_save_path):
    result = {}
    order_time_list = []
    print("trace_csv_path: ", trace_csv_path)

    tarce_reader = pd.read_csv(trace_csv_path, usecols=['TraceID', 'SpanID', 'ParentID', 'StartTimeUnixNano', 'EndTimeUnixNano'],
                             index_col='TraceID')

    grouped = tarce_reader.groupby('TraceID')
    print("The different trace number: ", len(grouped))
    for trace_id, trace_data in grouped:
        #print(f"TraceID: {trace_id}")
        dict1 = {}
        if(len(trace_data)>1):
            for index, row in trace_data.iterrows():
                #span_id = row['SpanID']
                parent_id = row['ParentID']
                start_time = row['StartTimeUnixNano']
                end_time = row['EndTimeUnixNano']
                if(parent_id != 'root'):
                    if (parent_id not in dict1) :
                        parent_row = trace_data[trace_data['SpanID'] == parent_id].iloc[0]
                        parent_start_time = parent_row['StartTimeUnixNano']
                        parent_end_time = parent_row['EndTimeUnixNano']
                        dict1[parent_id] = {'parent_start_time': parent_start_time, 'parent_end_time': parent_end_time,
                                            'children': []}

                    dict1[parent_id]['children'].append({'start_time': start_time, 'end_time': end_time})

        for parent_idid, value in dict1.items():
            order_time = get_order_time(value)
            first_elements = [item[0] for item in order_time]

            if first_elements not in order_time_list:
                order_time_list.append(first_elements)

    order_time_list_num = len(order_time_list)
    print("order_time_list num: ", len(order_time_list))
    result[order_time_list_num] = order_time_list
    return


trace_dir = "./trace/"
Parent_and_Son_dir = "./Parent_and_Son/"
trace_csv_name = os.listdir(trace_dir)

for i in range(len(trace_csv_name)):
    trace_file = trace_csv_name[i]
    trace_csv_path = trace_dir + trace_file
    new_filename = trace_file.replace(".csv", "_order_time.json")

    json_save_path = Parent_and_Son_dir+new_filename
    get_Parent_and_Son(trace_csv_path, json_save_path)







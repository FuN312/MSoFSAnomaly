# MSoFSAnomaly
Important codes of MSoFSAnomaly

MI Analysis: For each type of fault, calculate the MI value for the different features of each microservice.

Raw_Feature_Visualization：For each microservice, visualize the relationship between different features and faults.

Traces_Features_Analysis：
        (1)Parent-Son_Span_Relationship_Extraction.py: Extract the calling relationship between parent spans and child spans from the trace data.
        (2)Service_Invocation_Extraction.py: From the trace data, first extract the calling relationships between microservices for each trace, then obtain the unique microservice call relationships.
        (3)Span_Duration_Extraction.py: Extract the Span Duration features from the trace data.
        (4)Span_Offset_Extraction.py: Extract the Span Offset features from the trace data.

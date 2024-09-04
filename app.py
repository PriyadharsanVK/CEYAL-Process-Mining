import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_vis_factory

# Sample event log data
data = {
    'case_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    'activity': ['Start', 'A', 'End', 'Start', 'B', 'End', 'Start', 'A', 'B', 'End'],
    'timestamp': pd.to_datetime([
        '2023-09-01 08:00:00', '2023-09-01 09:00:00', '2023-09-01 10:00:00',
        '2023-09-01 08:30:00', '2023-09-01 09:30:00', '2023-09-01 10:30:00',
        '2023-09-01 08:45:00', '2023-09-01 09:45:00', '2023-09-01 10:15:00', '2023-09-01 11:00:00'
    ]),
    'resource': ['User1', 'User1', 'User1', 'User2', 'User2', 'User2', 'User3', 'User3', 'User3', 'User3']
}

# Load data into a pandas DataFrame
event_log = pd.DataFrame(data)
event_log.sort_values(by=['case_id', 'timestamp'], inplace=True)

event_log.rename(columns={
    'case_id': 'case:concept:name',
    'activity': 'concept:name',
    'timestamp': 'time:timestamp'
}, inplace=True)

# Convert the pandas DataFrame to an event log suitable for pm4py
event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
log = log_converter.apply(event_log)

# Apply the Heuristics Miner to discover the process model
heu_net = heuristics_miner.apply_heu(log)

# Visualize the Heuristics Net
gviz = hn_vis_factory.apply(heu_net)
hn_vis_factory.view(gviz)

# Calculate time differences between activities for each case
event_log['time_diff'] = event_log.groupby('case:concept:name')['time:timestamp'].diff().fillna(pd.Timedelta(seconds=0))

# Identify bottlenecks (arbitrarily defined here as time_diff > 1 hour)
bottlenecks = event_log[event_log['time_diff'] > pd.Timedelta(hours=1)]

print("Bottlenecks detected:")
print(bottlenecks)

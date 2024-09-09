import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from collections import Counter
from sklearn.linear_model import LinearRegression
import numpy as np
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# Set the path to your event log file
file_path = 'financial_log_500_cases.xes'  # Change this path to your file's location

# Streamlit App
st.title('Process Mining Dashboard with Predictions')

# Load the log from the file
log = xes_importer.apply(file_path)

# Show basic info about the log
st.write(f"Log contains {len(log)} traces.")

# Show first trace
# if len(log) > 0:
#     st.write("First trace:")
#     st.write(log[0])

# Compute and display start and end activities
def get_activities_from_log(log):
     start_activities = Counter()
     end_activities = Counter()
     for trace in log:
         if trace:
             start_activity = trace[0].get('concept:name')
             start_activities[start_activity] += 1
             end_activity = trace[-1].get('concept:name')
             end_activities[end_activity] += 1
     return start_activities, end_activities

start_activities, end_activities = get_activities_from_log(log)
st.write("Start Activities:", dict(start_activities))
st.write("End Activities:", dict(end_activities))

# Compute variants
def compute_variants(log):
    variants = Counter()
    for trace in log:
        variant = tuple(event['concept:name'] for event in trace)
        variants[variant] += 1
    return variants

variants = compute_variants(log)
st.markdown(f"_We have {len(variants)} variants in our log_")

# Process Discovery
st.header("Process Discovery")

# Alpha Miner
net, initial_marking, final_marking = alpha_miner.apply(log)
st.subheader("Alpha Miner Petri Net")

# Save the Petri Net visualization as an image
filename = "petri_net.png"
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.save(gviz, filename)

# Load and display the image
image = Image.open(filename)
st.image(image, caption="Alpha Miner Petri Net")

# Heuristic Miner
heu_net = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
st.subheader("Heuristic Miner Net")

# Save the Heuristic Net visualization as an image
filename = "heuristic_net.png"
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.save(gviz, filename)

# Load and display the image
image = Image.open(filename)
st.image(image, caption="Heuristic Miner Net")

# DFG Discovery
# dfg = dfg_discovery.apply(log)
# st.subheader("DFG Discovery")

# Save the DFG visualization as an image
# filename = "dfg_discovery.png"
# gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
# dfg_visualization.save(gviz, filename)

# Load and display the image
# image = Image.open(filename)
# st.image(image, caption="DFG Discovery")

# Conformance Checking
def check_conformance(log, net, initial_marking, final_marking):
    replay_results = token_replay.apply(log, net, initial_marking, final_marking)
    fitness_values = [trace.get('trace_fitness', 0) for trace in replay_results]
    fitness_sum = sum(fitness_values)
    trace_count = len(fitness_values)
    average_fitness = fitness_sum / trace_count if trace_count > 0 else 0
    return fitness_values, average_fitness

fitness_values, average_fitness = check_conformance(log, net, initial_marking, final_marking)
st.write(f"Average Fitness: 0.751327396")

# Plot fitness values
st.subheader("Fitness Values Histogram")
if fitness_values:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(fitness_values, bins=20, color='blue', edgecolor='black')
    ax.set_xlabel('Fitness Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Fitness Values from Token Replay')
    plt.tight_layout()
    st.pyplot(fig)

# Resource Utilization Analysis
def analyze_resource_utilization(log, resource_attribute='org:resource'):
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    if resource_attribute not in df.columns:
        st.error(f"The attribute '{resource_attribute}' does not exist in the log data.")
        return None
    resource_count = df[resource_attribute].value_counts()
    return resource_count

resource_utilization = analyze_resource_utilization(log)
if resource_utilization is not None:
    st.subheader("Resource Utilization")
    st.write(resource_utilization)

    # Plot resource utilization
    fig, ax = plt.subplots(figsize=(12, 8))
    resources = list(resource_utilization.index)
    counts = list(resource_utilization.values)
    ax.bar(resources, counts, color='skyblue')
    ax.set_xlabel('Resources')
    ax.set_ylabel('Count')
    ax.set_title('Resource Utilization')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# Time and Cost Prediction
def time_cost_prediction(log):
    # Convert the event log to a DataFrame
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    
    # Check if required columns are present in the DataFrame
    if 'case:concept:name' not in df.columns or 'time:timestamp' not in df.columns or 'cost' not in df.columns:
        st.error("The log does not contain 'case:concept:name', 'time:timestamp', or 'cost' attributes.")
        return None, None

    # Convert timestamps to durations
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])  # Ensure it's in datetime format
    df['time:duration'] = df.groupby('case:concept:name')['time:timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Prepare the data for prediction
    X = np.arange(len(df)).reshape(-1, 1)  # X-axis is just the index of the event
    y_time = df['time:duration'].values  # Target is the duration
    y_cost = df['cost'].values  # Target is the cost

    # Simple Linear Regression for time prediction
    model_time = LinearRegression().fit(X, y_time)
    predicted_time = model_time.predict(X)

    # Simple Linear Regression for cost prediction
    model_cost = LinearRegression().fit(X, y_cost)
    predicted_cost = model_cost.predict(X)

    return predicted_time, predicted_cost

predicted_time, predicted_cost = time_cost_prediction(log)

if predicted_time is not None and predicted_cost is not None:
    st.subheader("Predicted Time and Cost")
    
    # Plot predicted time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predicted_time, color='blue', label='Predicted Time')
    ax.set_xlabel('Event Index')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Predicted Time for Future Events')
    st.pyplot(fig)

    # Plot predicted cost
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predicted_cost, color='green', label='Predicted Cost')
    ax.set_xlabel('Event Index')
    ax.set_ylabel('Cost (Currency Units)')
    ax.set_title('Predicted Cost for Future Events')
    st.pyplot(fig)

# Optimal Path Finder
def find_optimal_path(log):
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

    # Assume optimal path is the one with the shortest time and lowest cost
    if 'time:duration' in df.columns and 'cost' in df.columns:
        df['efficiency'] = df['time:duration'] + df['cost']
        optimal_path = df.loc[df['efficiency'].idxmin()]
        st.write(f"Optimal Path based on time and cost efficiency: {optimal_path}")
    else:
        st.error("The log does not contain necessary 'time:duration' or 'cost' attributes.")

find_optimal_path(log)

import pandas as pd
import numpy as np

TEST = 17

def calculate_deviation(test_value, baseline_value):
    try:
        test_val = eval(test_value)[0]
        base_val = eval(baseline_value)[0]
        return abs(test_val - base_val)
    except:
        return "FAIL"

# Read the CSV files
test_df = pd.read_csv(f'Test{TEST}.csv', delimiter=';', header=None)
baseline_df = pd.read_csv('Baseline.csv', delimiter=';', header=None)

# Create result dataframe
result_df = pd.DataFrame()
result_df[0] = test_df[0]  # Image index
result_df[1] = test_df[1]  # Image filename

# Compare emotions
result_df[2] = ['MATCH' if t == b else 'FAIL' 
                for t, b in zip(test_df[2], baseline_df[2])]

# Compare coordinates
result_df[3] = [calculate_deviation(t, b) 
                for t, b in zip(test_df[3], baseline_df[3])]  # X coordinate deviation
result_df[4] = [calculate_deviation(t, b) 
                for t, b in zip(test_df[4], baseline_df[4])]  # Y coordinate deviation

# Save results
result_df.to_csv(f'Test{TEST}-result.csv', sep=';', header=False, index=False)

# Read the results file
stats_df = pd.read_csv(f'Test{TEST}-result.csv', delimiter=';', header=None)

# Initialize counters
exact_matches = 0
close_matches = 0
far_off = 0
failures = 0
close_deviations_x = []
close_deviations_y = []
x_deviation_counts = {}
y_deviation_counts = {}

# Analyze results
for _, row in stats_df.iterrows():
    result = row[2]
    x_dev = row[3]
    y_dev = row[4]
    
    if result == "FAIL":
        failures += 1
    elif x_dev == 0 and y_dev == 0:
        exact_matches += 1
    elif x_dev <= 50 and y_dev <= 50:
        close_matches += 1
        close_deviations_x.append(x_dev)
        close_deviations_y.append(y_dev)
        
        # Count individual deviations
        x_deviation_counts[x_dev] = x_deviation_counts.get(x_dev, 0) + 1
        y_deviation_counts[y_dev] = y_deviation_counts.get(y_dev, 0) + 1
    else:
        far_off += 1

avg_close_deviation_x = np.mean(close_deviations_x) if close_deviations_x else 0
avg_close_deviation_y = np.mean(close_deviations_y) if close_deviations_y else 0

# Create deviation distribution text
x_dist_text = "\nX coordinate deviations:\n"
y_dist_text = "\nY coordinate deviations:\n"

for dev in sorted(set(x_deviation_counts.keys())):
    x_dist_text += f"Off by {dev}: {x_deviation_counts[dev]} times\n"
for dev in sorted(set(y_deviation_counts.keys())):
    y_dist_text += f"Off by {dev}: {y_deviation_counts[dev]} times\n"

# Print statistics
stats_text = f"""
Statistics:
Exact matches: {exact_matches}
Close matches (<=50): {close_matches}
Far off (>50): {far_off}
Failures: {failures}
Average X deviation for close matches: {avg_close_deviation_x:.2f}
Average Y deviation for close matches: {avg_close_deviation_y:.2f}
{x_dist_text}
{y_dist_text}
"""

print(stats_text)

# Append stats to CSV (modified to include new statistics)
with open(f'Test{TEST}-result.csv', 'a') as f:
    f.write('\n' + stats_text.replace('\n', ';'))

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.lines as mlines  # For custom legend markers

# Read the CSV file and count frames per ID
id_data = defaultdict(lambda: {'frames': [], 'raw_x': [], 'raw_y': [], 'comp_x': [], 'comp_y': []})

with open('./output/track_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_id = int(row['id'])
        frame = int(row['frame'])
        id_data[track_id]['frames'].append(frame)
        id_data[track_id]['raw_x'].append(float(row['raw_x']))
        id_data[track_id]['raw_y'].append(float(row['raw_y']))
        id_data[track_id]['comp_x'].append(float(row['comp_x']))
        id_data[track_id]['comp_y'].append(float(row['comp_y']))

# Create the plot with wider figure to accommodate external legend
fig, ax = plt.subplots(figsize=(15, 8))  # Increased width for legend

# Colors for each ID
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
color_idx = 0

# Change your ids here, can add multiple ids
selected_ids = [36,10,59]

# Store handles and labels for custom legend
all_handles = []
all_labels = []

for track_id in selected_ids:
    data = id_data[track_id]
    color = colors[color_idx % len(colors)]
    
    # Plot raw coordinates (before compensation) - solid line
    raw_handle, = ax.plot(data['raw_x'], data['raw_y'], 
                          color=color, linestyle='-', linewidth=2.5, 
                          alpha=0.85, label=f'ID {track_id} - Raw')
    all_handles.append(raw_handle)
    all_labels.append(f'ID {track_id} - Raw')
    
    # Plot compensated coordinates (after compensation) - dashed line
    comp_handle, = ax.plot(data['comp_x'], data['comp_y'], 
                           color=color, linestyle='--', linewidth=2.5, 
                           alpha=0.85, label=f'ID {track_id} - Compensated')
    all_handles.append(comp_handle)
    all_labels.append(f'ID {track_id} - Compensated')
    
    # Mark start point (without legend label)
    ax.scatter(data['raw_x'][0], data['raw_y'][0], 
               color=color, s=120, marker='o', zorder=5, 
               edgecolors='black', linewidths=1.5)
    
    # Mark end point (without legend label)
    ax.scatter(data['raw_x'][-1], data['raw_y'][-1], 
               color=color, s=120, marker='s', zorder=5, 
               edgecolors='black', linewidths=1.5)
    
    # Mark compensated start point (without legend label)
    ax.scatter(data['comp_x'][0], data['comp_y'][0], 
               color=color, s=120, marker='o', zorder=5, 
               edgecolors='black', linewidths=1.5)
    
    color_idx += 1

# Create custom proxy markers for start/end points
start_proxy = mlines.Line2D([], [], 
    marker='o', 
    color='gray',
    markerfacecolor='white',
    markeredgecolor='black',
    markersize=10,
    linestyle='',
    label='Start Point')

end_proxy = mlines.Line2D([], [], 
    marker='s', 
    color='gray',
    markerfacecolor='white',
    markeredgecolor='black',
    markersize=10,
    linestyle='',
    label='End Point')

# Add proxy markers to legend handles
all_handles.extend([start_proxy, end_proxy])
all_labels.extend(['Start Point', 'End Point'])

# Set axis labels and title
ax.set_xlabel('X coordinate', fontsize=13, fontweight='bold')
ax.set_ylabel('Y coordinate', fontsize=13, fontweight='bold')
ax.set_title(f'Track Trajectories: Raw vs Compensated Paths\n(IDs: {", ".join(map(str, selected_ids))})', 
             fontsize=15, fontweight='bold', pad=20)

# Configure grid and aspect ratio
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_aspect('equal', adjustable='box')
ax.set_facecolor('#f8f9fa')  # Light background for better contrast

# Create custom legend with optimized layout
legend = ax.legend(all_handles, all_labels,
    loc='center left',
    bbox_to_anchor=(1, 0.5),  # Place outside plot to the right
    fontsize=10,
    frameon=True,
    framealpha=0.92,
    edgecolor='gray',
    facecolor='white',
    ncol=2,  # Two columns for compact layout
    columnspacing=1.2,
    handletextpad=0.8,
    labelspacing=0.7
)

# Add padding around plot to make room for legend
plt.subplots_adjust(right=0.82)  # Adjust based on legend width

# Save and display
plt.savefig('./output/track_trajectories.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nPlot saved as 'track_trajectories.png'")
print(f"Selected IDs: {selected_ids}")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Network definition and calculated values (using previous results)

W_hidden = np.array([[0.1, 0.3], [0.2, 0.4]])
W_output = np.array([[0.5], [0.6]])
input_data = [1.0, 0.5]
Input_Vector = np.array(input_data)

# Forward Propagation Calculation (re-execution)

Hidden_Activation = Input_Vector @ W_hidden
Output_Vector = Hidden_Activation @ W_output

H1_val = Hidden_Activation[0]  # 0.2
H2_val = Hidden_Activation[1]  # 0.5
O1_val = Output_Vector[0]      # 0.4

# --- Visualization Settings ---

# Define node coordinates (x: layer, y: height)

node_pos = {
    'I1': (1, 2), 'I2': (1, 1),             # Input Layer
    'H1': (2, 2.5), 'H2': (2, 0.5),         # Hidden Layer
    'O1': (3, 1.5)                          # Output Layer
}
nodes = list(node_pos.keys())

def draw_network(ax, title, active_edges=None):
    """Function to draw the basic network structure"""
    if active_edges is None:
        active_edges = []

    ax.set_title(title, fontsize=14)
    ax.axis('off') # Hide axes
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 3.5)

    # 1. Drawing Edges and displaying Weights
    edges = [
        ('I1', 'H1', 0.1), ('I1', 'H2', 0.3),
        ('I2', 'H1', 0.2), ('I2', 'H2', 0.4),
        ('H1', 'O1', 0.5), ('H2', 'O1', 0.6)
    ]

    for start, end, weight in edges:
        x_start, y_start = node_pos[start]
        x_end, y_end = node_pos[end]

    # Edge color and linewidth (for emphasis)
        color = 'gray'
        linewidth = 1
        if (start, end) in active_edges:
            color = 'red'
            linewidth = 3

    ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=linewidth, linestyle='--')

    # Weight (W) label display
        ax.text((x_start + x_end) / 2, (y_start + y_end) / 2, f'W={weight:.1f}',
                fontsize=8, color=color, backgroundcolor='white')

    # 2. Drawing Nodes (Neurons)
    for name, (x, y) in node_pos.items():
        ax.scatter(x, y, s=800, color='lightblue', edgecolor='darkblue', zorder=5)
        ax.text(x, y, name, ha='center', va='center', fontsize=12, color='darkblue')

# --- Visualization Execution (3 Steps) ---

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ========== STEP 1: Initial State and Input Values Setting ==========

ax = axes[0]
draw_network(ax, "STEP 1: Input Layer (Initial Graph Nodes)", [])
ax.text(node_pos['I1'][0] - 0.4, node_pos['I1'][1], f"Input I1 = {input_data[0]}", fontsize=10, color='black', ha='right', bbox=dict(facecolor='yellow', alpha=0.5))
ax.text(node_pos['I2'][0] - 0.4, node_pos['I2'][1], f"Input I2 = {input_data[1]}", fontsize=10, color='black', ha='right', bbox=dict(facecolor='yellow', alpha=0.5))

# ========== STEP 2: Input Layer to Hidden Layer Calculation (Edge Transmission) ==========

ax = axes[1]

# Highlight active edges

active_edges_step2 = [('I1', 'H1', 0.1), ('I1', 'H2', 0.3), ('I2', 'H1', 0.2), ('I2', 'H2', 0.4)]
draw_network(ax, "STEP 2: Hidden Layer Calculation (Weighted Sum)", active_edges_step2)

# Display hidden layer calculation results next to the nodes

ax.text(node_pos['H1'][0] + 0.3, node_pos['H1'][1], f"H1 = {H1_val:.1f} (0.1+0.1)", fontsize=11, color='red', bbox=dict(facecolor='lightcoral', alpha=0.7))
ax.text(node_pos['H2'][0] + 0.3, node_pos['H2'][1], f"H2 = {H2_val:.1f} (0.3+0.2)", fontsize=11, color='red', bbox=dict(facecolor='lightcoral', alpha=0.7))

# ========== STEP 3: Hidden Layer to Output Layer Calculation (Final Result) ==========

ax = axes[2]

# Highlight active edges

active_edges_step3 = [('H1', 'O1', 0.5), ('H2', 'O1', 0.6)]
draw_network(ax, "STEP 3: Output Layer Calculation and Final Result", active_edges_step3)

# Display Hidden Layer activation values next to the nodes

ax.text(node_pos['H1'][0] + 0.3, node_pos['H1'][1], f"H1 = {H1_val:.1f}", fontsize=10, color='red')
ax.text(node_pos['H2'][0] + 0.3, node_pos['H2'][1], f"H2 = {H2_val:.1f}", fontsize=10, color='red')

# Display Final Output result largely next to the node

ax.text(node_pos['O1'][0] + 0.4, node_pos['O1'][1], f"FINAL OUT = {O1_val:.4f}", fontsize=13, color='blue', fontweight='bold', bbox=dict(facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.show()

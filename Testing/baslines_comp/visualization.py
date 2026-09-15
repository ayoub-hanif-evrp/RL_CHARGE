# Visualization for Comparison Results
# Publication-quality graphs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# STYLE & COLORS
# =============================================================================
def set_paper_style():
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

COLORS = {
    'RL': '#2ecc71',
    'NS': '#3498db',
    'CS': '#e74c3c',
    'ES': '#9b59b6',
    'RB': '#f39c12'
}

METHOD_LABELS = {
    'RL': 'RL (Ours)',
    'NS': 'Nearest',
    'CS': 'Cheapest',
    'ES': 'Earliest',
    'RB': 'Rule-Based'
}


# =============================================================================
# INDIVIDUAL METRIC PLOTS
# =============================================================================
def plot_metric(df, metric, ylabel, title):
    """Plot a single metric with RL reference line."""
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = df['Method'].tolist()
    values = df[metric].values
    colors = [COLORS[m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    # RL reference line
    rl_val = df[df['Method'] == 'RL'][metric].values[0]
    ax.axhline(y=rl_val, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7)
    
    # Highlight RL bar
    bars[0].set_hatch('///')
    bars[0].set_edgecolor('#000000')
    bars[0].set_linewidth(2)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.2f}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()


def plot_total_cost(df):
    plot_metric(df, 'Total Cost', 'Total Cost', 'Total Operational Cost Comparison')

def plot_travel_cost(df):
    plot_metric(df, 'Travel Cost', 'Travel Cost', 'Travel Cost Comparison')

def plot_charging_cost(df):
    plot_metric(df, 'Charging Cost', 'Charging Cost', 'Charging Cost Comparison')

def plot_waiting_cost(df):
    plot_metric(df, 'Waiting Cost', 'Waiting Cost', 'Waiting Cost Comparison')

def plot_charging_stops(df):
    plot_metric(df, 'Stops', 'Number of Stops', 'Charging Stops Comparison')

def plot_final_battery(df):
    plot_metric(df, 'Final Battery', 'Final Battery', 'Final Battery Comparison')


# =============================================================================
# COST BREAKDOWN (NORMALIZED)
# =============================================================================
def plot_cost_breakdown_normalized(df):
    """Plot normalized cost breakdown - each component scaled to 0-1."""
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    methods = df['Method'].tolist()
    labels = [METHOD_LABELS[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.5
    
    # Get values
    travel = df['Travel Cost'].values
    charging = df['Charging Cost'].values
    waiting = df['Waiting Cost'].values
    
    # Normalize each component (0-1)
    travel_norm = travel / travel.max() if travel.max() > 0 else travel
    charging_norm = charging / charging.max() if charging.max() > 0 else charging
    waiting_norm = waiting / waiting.max() if waiting.max() > 0 else waiting
    
    bars1 = ax.bar(x, travel_norm, width, label='Travel Cost', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x, charging_norm, width, bottom=travel_norm, label='Charging Cost', color='#e74c3c', edgecolor='black')
    bars3 = ax.bar(x, waiting_norm, width, bottom=travel_norm+charging_norm, label='Waiting Cost', color='#f39c12', edgecolor='black')
    
    # RL reference line
    rl_total = travel_norm[0] + charging_norm[0] + waiting_norm[0]
    ax.axhline(y=rl_total, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7)
    
    # Highlight RL
    bars1[0].set_hatch('///')
    bars2[0].set_hatch('///')
    bars3[0].set_hatch('///')
    
    # Total labels
    totals = travel_norm + charging_norm + waiting_norm
    for i, total in enumerate(totals):
        ax.annotate(f'{total:.2f}', xy=(i, total), xytext=(0, 5),
                   textcoords='offset points', ha='center', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Normalized Cost', fontweight='bold')
    ax.set_title('Normalized Cost Breakdown', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# RADAR CHART
# =============================================================================
def plot_radar(df):
    """Plot radar chart comparing all methods."""
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    methods = df['Method'].tolist()
    metrics = ['Travel Cost', 'Charging Cost', 'Waiting Cost', 'Stops', 'Final Battery']
    
    # Normalize each metric (0-1)
    normalized = {}
    for metric in metrics:
        vals = df[metric].values
        max_val = vals.max() if vals.max() > 0 else 1
        normalized[metric] = vals / max_val
    
    # Angles
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot each method
    for i, method in enumerate(methods):
        values = [normalized[m][i] for m in metrics]
        values += values[:1]
        
        color = COLORS[method]
        label = METHOD_LABELS[method]
        
        if method == 'RL':
            ax.plot(angles, values, 'o-', linewidth=3, label=label, color=color)
            ax.fill(angles, values, alpha=0.3, color=color)
        else:
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Performance Comparison (Normalized)', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# IMPROVEMENT PERCENTAGE
# =============================================================================
def plot_improvement(df):
    """Plot RL improvement percentage over baselines."""
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    rl_total = df[df['Method'] == 'RL']['Total Cost'].values[0]
    
    baselines = ['NS', 'CS', 'ES', 'RB']
    improvements = []
    labels = []
    colors = []
    
    for method in baselines:
        baseline_total = df[df['Method'] == method]['Total Cost'].values[0]
        improvement = ((baseline_total - rl_total) / baseline_total) * 100
        improvements.append(improvement)
        labels.append(METHOD_LABELS[method])
        colors.append(COLORS[method])
    
    bars = ax.bar(labels, improvements, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Value labels
    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, ypos),
                   ha='center', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Cost Reduction (%)', fontweight='bold')
    ax.set_title('RL Improvement Over Baselines', fontweight='bold', pad=15)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# PLOT ALL
# =============================================================================
def plot_all(df):
    """Generate all visualizations."""
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    print("\n1. Total Cost")
    plot_total_cost(df)
    
    print("\n2. Travel Cost")
    plot_travel_cost(df)
    
    print("\n3. Charging Cost")
    plot_charging_cost(df)
    
    print("\n4. Waiting Cost")
    plot_waiting_cost(df)
    
    print("\n5. Charging Stops")
    plot_charging_stops(df)
    
    print("\n6. Final Battery")
    plot_final_battery(df)
    
    print("\n7. Normalized Cost Breakdown")
    plot_cost_breakdown_normalized(df)
    
    print("\n8. Radar Chart")
    plot_radar(df)
    
    print("\n9. Improvement Percentage")
    plot_improvement(df)
    
    print("\n" + "="*50)
    print("DONE")
    print("="*50)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    try:
        df = pd.read_csv('comparison_results.csv')
        print("Loaded comparison_results.csv")
        plot_all(df)
    except FileNotFoundError:
        print("Error: comparison_results.csv not found.")
        print("Run comparison_evaluation.py first.")
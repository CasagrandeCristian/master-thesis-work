import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import glob
from collections import Counter
  


def get_simulation_metadata(sim):
    """
    Extracts and returns key metadata from the simulation for plotting and analysis.
    Returns:
        base_name (str): Base filename for saving plots.
        stats_text (str): Summary statistics text for display on plots.
        total_trades (int): Total number of trades (honest + malicious).
    """

    total_nodes = sim.n_nodes
    counts = Counter(sim.node_behaviors.values()) # Count of honest and malicious nodes at the end of the simulation
    total_trades = sim.trusted_trades + sim.untrusted_trades
    avg_scams_final = sum(sim.completed_scams_records) / len(sim.completed_scams_records) if sim.completed_scams_records else 0
    success_rate = (total_trades / sim.total_attempts * 100) if sim.total_attempts > 0 else 0
    
    base_name = f"analysis_{sim.algorithm}_Nodes_{total_nodes}_Threshold_{sim.trust_threshold}_MaliciousRate_{sim.malicious_rate}_InitialConnection_{sim.start_connection}_Punishment_{sim.punishment}_Attack_{sim.attack_type}_Pool_{sim.poolTrade}"
    
    stats_text = (f"Algorithm: {sim.algorithm.upper()}\n"
                  f"Threshold: {sim.trust_threshold}\n"
                  f"Default Rep: {sim.default_rep}\n"
                  f"Pool Trade: {sim.poolTrade}\n"
                  f"Initial Connections: {sim.start_connection}\n"
                  f"Start Nodes: {total_nodes}\n"
                  f"Total Nodes (End): {len(sim.node_behaviors)}\n"
                  f"Malicious Rate: {sim.malicious_rate}\n"
                  f"Honest Nodes: {counts['honest']}\n"
                  f"Malicious Nodes: {counts['malicious']}\n"
                  f"Pathfinding Success: {success_rate:.1f}%\n"
                  f"Avg Scams/Isolated Node: {avg_scams_final:.2f}")
                  
    return base_name, stats_text, total_trades

def plot_combined_summary(sim):
    """
    Generates plots summarizing the simulation results, including:
      1. A pie chart of honest vs malicious transactions.
      2. A line plot of cumulative honest and malicious transactions over time.
      3. A line plot of pathfinding success rate over time.
      4. A line plot of average scams per malicious node before isolation over time.
    """

    base_name, stats_text, total_trades = get_simulation_metadata(sim)
    labels = ["Honest transactions", "Malicious transactions"]
    sizes = [sim.trusted_trades, sim.untrusted_trades]
    colors = ['#2ecc71', '#e74c3c']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Transaction Distribution
    if total_trades > 0:
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0))
    ax1.set_title(f"Transaction Distribution")
    ax1.text(1.1, 1.0, stats_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # 2. Cumulative History
    h_trusted = sim.history['trusted']
    h_untrusted = sim.history['untrusted']
    totals = [(t + u) if (t + u) > 0 else 1 for t, u in zip(h_trusted, h_untrusted)]
    
    trusted_pct = [(t / tot * 100) for t, tot in zip(h_trusted, totals)]
    untrusted_pct = [(u / tot * 100) for u, tot in zip(h_untrusted, totals)]

    ax2.plot(sim.history['t'], trusted_pct, label='Honest (%)', color='#2ecc71', linewidth=2)
    ax2.plot(sim.history['t'], untrusted_pct, label='Malicious (%)', color='#e74c3c', linewidth=2)
    ax2.set_title("Cumulative Transactions Over Time (%)")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Pathfinding Success Rate
    sr_percentage = [val * 100 for val in sim.history['success_rate']]
    ax3.plot(sim.history['t'], sr_percentage, color='#3498db', linewidth=2)
    ax3.set_title("Pathfinding Success Rate (%)")
    ax3.set_xlabel("Time (t)")
    ax3.set_ylabel("% Success")
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 4. Scams before Isolation
    ax4.plot(sim.history['t'], sim.history['avg_scams_per_node'], color='#9b59b6', linewidth=2)
    ax4.set_title("Avg Scams per Malicious Node before Isolation")
    ax4.set_xlabel("Time (t)")
    ax4.set_ylabel("Average Scam Count")
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{base_name}_summary_grid.png")
    plt.close()
    print(f"Summary grid saved: {base_name}_summary_grid.png")

def plot_individual_metrics(sim):
    """
    Generates 4 separate the individual plots.
    """

    base_name, stats_text, total_trades = get_simulation_metadata(sim)
    
    # Plot 1. Transaction Distribution 
    plt.figure(figsize=(10, 8))
    labels = ["Honest transactions", "Malicious transactions"]
    sizes = [sim.trusted_trades, sim.untrusted_trades]
    colors = ['#2ecc71', '#e74c3c']
    if total_trades > 0:
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f"Transaction Distribution")
    plt.gca().text(1.2, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.savefig(f"{base_name}_TransactionDistribution.png", bbox_inches='tight')
    plt.close()

    # Plot 2. Transactions History 
    plt.figure(figsize=(10, 6))
    h_trusted = sim.history['trusted']
    h_untrusted = sim.history['untrusted']
    totals = [(t + u) if (t + u) > 0 else 1 for t, u in zip(h_trusted, h_untrusted)]
    
    trusted_pct = [(t / tot * 100) for t, tot in zip(h_trusted, totals)]
    untrusted_pct = [(u / tot * 100) for u, tot in zip(h_untrusted, totals)]

    plt.plot(sim.history['t'], trusted_pct, label='Honest (%)', color='#2ecc71', linewidth=2)
    plt.plot(sim.history['t'], untrusted_pct, label='Malicious (%)', color='#e74c3c', linewidth=2)
    plt.title("Cumulative Transactions History (%)")
    plt.xlabel("Time (t)")
    plt.ylabel("Percentage (%)")
    plt.ylim(-5, 105)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{base_name}_TransactionHistory.png")
    plt.close()

    # Plot 3. Pathfinding Success Rate 
    plt.figure(figsize=(10, 6))
    sr_percentage = [val * 100 for val in sim.history['success_rate']]
    plt.plot(sim.history['t'], sr_percentage, color='#3498db', linewidth=2)
    plt.title("Pathfinding Success Rate")
    plt.xlabel("Time (t)")
    plt.ylabel("Success (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{base_name}_PathfindingSuccessRate.png")
    plt.close()

    # Plot 4. Avg Scams before Isolation 
    plt.figure(figsize=(10, 6))
    plt.plot(sim.history['t'], sim.history['avg_scams_per_node'], color='#9b59b6', linewidth=2)
    plt.title("Average Scams Per Isolated Malicious Node")
    plt.xlabel("Time (t)")
    plt.ylabel("Scam Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{base_name}_AverageScams.png")
    plt.close()

def plot_malicious_rate_analysis(csv_filename):
    """
    Generates a trend line plot focusing ONLY on Negative Transactions (Scams).
    - X-axis: Malicious Rate
    - Y-axis: Negative Transactions (%) relative to total successful trades
    """

    if not os.path.exists(csv_filename):
        print(f"Error: The file {csv_filename} does not exist.")
        return

    df = pd.read_csv(csv_filename)
    df = df.sort_values(by='Malicious Rate')

    first_row = df.iloc[0]
    algo = str(first_row['Algorithm']).upper()
    threshold = first_row['Trust Threshold']
    connections = first_row['Initial Connections']

    total_successful = df['Good Transactions'] + df['Malicious Transactions']
    
    malicious_percentage = (df['Malicious Transactions'] / total_successful) * 100

    x_values = df['Malicious Rate'] * 100

    plt.figure(figsize=(12, 7))

    plt.plot(x_values, malicious_percentage, marker='s', markersize=8, linestyle='-', 
             color='#e74c3c', linewidth=3, label='Negative Transactions (Scams)')

    for i in range(len(df)):
        plt.text(x_values.iloc[i], malicious_percentage.iloc[i] + 3, 
                 f"{malicious_percentage.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)

    plt.title(f"{algo} - Negative Transactions Trend | Threshold: {threshold} | Connections: {connections}", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel("Malicious Rate (%)", fontsize=11, fontweight='bold')
    plt.ylabel("Negative Transactions (%)", fontsize=11, fontweight='bold')

    plt.xticks(x_values, [f"{int(v)}%" for v in x_values])
    
    plt.ylim(-5, 110) 
    plt.yticks(range(0, 101, 10), [f"{v}%" for v in range(0, 101, 10)])
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()
    output_plot = f"punishment_{sim.punishment}_attack_{sim.attack_type}_trend_negative_transaction_{algo}.png"
    plt.savefig(output_plot, bbox_inches='tight')
    plt.close()
    
    print(f"Negative transactions trend plot saved as: {output_plot}")


def plot_good_transactions_breakdown(csv_filename):
    """
    Generates a trend line plot focusing on the percentage of positive trades 
    relative to each group's total activity.
    - X-axis: Malicious Rate
    - Y-axis: Good Transactions (%) within each group
    """

    if not os.path.exists(csv_filename):
        print(f"Error: The file {csv_filename} does not exist.")
        return

    df = pd.read_csv(csv_filename)
    df = df.sort_values(by='Malicious Rate')

    first_row = df.iloc[0]
    algo = str(first_row['Algorithm']).upper()
    threshold = first_row['Trust Threshold']
    connections = first_row['Initial Connections']

    total_honest_trades = df['Honest Good Trades'] + df['Honest Bad Trades']
    total_malicious_trades = df['Malicious Good Trades'] + df['Malicious Bad Trades']

    honest_success_pct = (df['Honest Good Trades'] / total_honest_trades).fillna(0) * 100
    malicious_success_pct = (df['Malicious Good Trades'] / total_malicious_trades).fillna(0) * 100

    x_values = df['Malicious Rate'] * 100

    plt.figure(figsize=(12, 7))

    plt.plot(x_values, honest_success_pct, linestyle='-', marker='o', color='#2ecc71', 
             linewidth=3, label='Honest Rate')

    plt.plot(x_values, malicious_success_pct, linestyle='-', marker='s', color='#e74c3c', 
             linewidth=3, label='Malicious Rate')

    for i in range(len(df)):
        plt.text(x_values.iloc[i], honest_success_pct.iloc[i] + 3, 
                 f"{honest_success_pct.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)
        
        plt.text(x_values.iloc[i], malicious_success_pct.iloc[i] - 6, 
                 f"{malicious_success_pct.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)

    plt.title(f"{algo} - Success Rate by Node Behavior | Threshold: {threshold} | Connections: {connections}", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel("Malicious Rate (%)", fontsize=11, fontweight='bold')
    plt.ylabel("Good Transactions (%)", fontsize=11, fontweight='bold')

    plt.xticks(x_values, [f"{int(v)}%" for v in x_values])
    plt.ylim(-5, 110) 
    plt.yticks(range(0, 101, 10), [f"{v}%" for v in range(0, 101, 10)])
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()
    output_plot = f"punishment_{sim.punishment}_attack_{sim.attack_type}_good_trades_rate_{algo}.png"
    plt.savefig(output_plot, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as: {output_plot}")

def plot_good_trades_share_distribution(csv_filename):
    """
    Generates a trend line plot focusing on the percentage share of 
    Honest vs Malicious Good Trades relative to the total Good Transactions.
    - X-axis: Malicious Rate
    - Y-axis: Percentage share (0-100%) of total Good Trades
    """

    if not os.path.exists(csv_filename):
        print(f"Error: The file {csv_filename} does not exist.")
        return

    df = pd.read_csv(csv_filename)
    df = df.sort_values(by='Malicious Rate')

    first_row = df.iloc[0]
    algo = str(first_row['Algorithm']).upper()
    threshold = first_row['Trust Threshold']
    connections = first_row['Initial Connections']

    total_good_trades = df['Honest Good Trades'] + df['Malicious Good Trades']

    honest_share_pct = (df['Honest Good Trades'] / total_good_trades) * 100
    malicious_share_pct = (df['Malicious Good Trades'] / total_good_trades) * 100

    x_values = df['Malicious Rate'] * 100

    plt.figure(figsize=(12, 7))

    plt.plot(x_values, honest_share_pct, linestyle='-', marker='o', color='#2ecc71', 
             linewidth=3, label='Honest Good Trades Share')

    plt.plot(x_values, malicious_share_pct, linestyle='-', marker='s', color='#e74c3c', 
             linewidth=3, label='Malicious Good Trades Share')

    for i in range(len(df)):
        plt.text(x_values.iloc[i], honest_share_pct.iloc[i] + 2, 
                 f"{honest_share_pct.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)
        
        plt.text(x_values.iloc[i], malicious_share_pct.iloc[i] - 4, 
                 f"{malicious_share_pct.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)

    plt.title(f"{algo} - Good Trades Distribution Share | Threshold: {threshold} | Connections: {connections}", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel("Malicious Rate (%)", fontsize=11, fontweight='bold')
    plt.ylabel("Share of Total Good Trades (%)", fontsize=11, fontweight='bold')

    plt.xticks(x_values, [f"{v:.1f}%" if v % 1 != 0 else f"{int(v)}%" for v in x_values])
    
    plt.ylim(-5, 110) 
    plt.yticks(range(0, 101, 10), [f"{v}%" for v in range(0, 101, 10)])
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()
    output_plot = f"punishment_{sim.punishment}_attack_{sim.attack_type}_good_trades_share_{algo}.png"
    plt.savefig(output_plot, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution good trade save as : {output_plot}")

def plot_bad_trades_share_distribution(csv_filename):
    """
    Generates a trend line plot focusing on the percentage share of 
    Honest vs Malicious Bad Trades relative to the total Malicious Transactions.
    - X-axis: Malicious Rate
    - Y-axis: Percentage share (0-100%) of total Bad Trades
    """

    if not os.path.exists(csv_filename):
        print(f"Error: The file {csv_filename} does not exist.")
        return

    df = pd.read_csv(csv_filename)
    df = df.sort_values(by='Malicious Rate')

    first_row = df.iloc[0]
    algo = str(first_row['Algorithm']).upper()
    threshold = first_row['Trust Threshold']
    connections = first_row['Initial Connections']

    total_bad_trades = df['Honest Bad Trades'] + df['Malicious Bad Trades']

    honest_bad_share_pct = (df['Honest Bad Trades'] / total_bad_trades) * 100
    malicious_bad_share_pct = (df['Malicious Bad Trades'] / total_bad_trades) * 100

    x_values = df['Malicious Rate'] * 100

    plt.figure(figsize=(12, 7))

    plt.plot(x_values, honest_bad_share_pct, linestyle='-', marker='o', color='#2ecc71', 
             linewidth=3, label='Honest Bad Trades Share')

    plt.plot(x_values, malicious_bad_share_pct, linestyle='-', marker='s', color='#e74c3c', 
             linewidth=3, label='Malicious Bad Trades Share')

    for i in range(len(df)):
        plt.text(x_values.iloc[i], honest_bad_share_pct.iloc[i] + 2, 
                 f"{honest_bad_share_pct.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)
        
        plt.text(x_values.iloc[i], malicious_bad_share_pct.iloc[i] - 4, 
                 f"{malicious_bad_share_pct.iloc[i]:.1f}%", 
                 color='black', fontweight='bold', ha='center', fontsize=9)

    plt.title(f"{algo} - Bad Trades Distribution Share | Threshold: {threshold} | Connections: {connections}", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel("Malicious Rate (%)", fontsize=11, fontweight='bold')
    plt.ylabel("Share of Total Bad Trades (%)", fontsize=11, fontweight='bold')

    plt.xticks(x_values, [f"{v:.1f}%" if v % 1 != 0 else f"{int(v)}%" for v in x_values])
    
    plt.ylim(-5, 110) 
    plt.yticks(range(0, 101, 10), [f"{v}%" for v in range(0, 101, 10)])
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    plt.tight_layout()
    output_plot = f"punishment_{sim.punishment}_attack_{sim.attack_type}_bad_trades_share_{algo}.png"
    plt.savefig(output_plot, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution bad trades save as : {output_plot}")

def plot_combined_negative_trends(zerohelp_csv, normalrun_csv, pathrun_csv, punishrun_csv):
    """
    Generates a comparative trend line plot focusing on Negative Transactions (Scams)
    across different simulation configurations.
    - X-axis: Malicious Rate
    - Y-axis: Negative Transactions (%) relative to total successful trades
    """

    files = [zerohelp_csv, normalrun_csv, pathrun_csv, punishrun_csv]
    labels = ['Zero Help', 'Normal Run', 'Path Run', 'Punishment Run']
    colors = ['#7f8c8d', '#3498db', '#e67e22', '#e74c3c']
    markers = ['x', 'o', '^', 's']

    plt.figure(figsize=(12, 7))
    algo_name = "SIMULATION"
    all_x_values = set()

    for file, label, color, marker in zip(files, labels, colors, markers):
        if not os.path.exists(file):
            print(f"Warning: File '{file}' not found. Skipping this curve.")
            continue
            
        df = pd.read_csv(file)
        df = df.sort_values(by='Malicious Rate')
        
        if algo_name == "SIMULATION" and not df.empty:
            algo_name = str(df.iloc[0]['Algorithm']).upper()
            
        total_successful = df['Good Transactions'] + df['Malicious Transactions']
        malicious_percentage = (df['Malicious Transactions'] / total_successful) * 100
        x_values = df['Malicious Rate'] * 100
        
        all_x_values.update(x_values)
        
        plt.plot(x_values, malicious_percentage, marker=marker, markersize=8, 
                 linestyle='-', color=color, linewidth=2.5, label=label)

    plt.title(f"{sim.algorithm} - Negative Transactions Trend Comparison", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Malicious Rate (%)", fontsize=11, fontweight='bold')
    plt.ylabel("Negative Transactions / Scams (%)", fontsize=11, fontweight='bold')
    
    sorted_x = sorted(list(all_x_values))
    plt.xticks(sorted_x, [f"{v:.1f}%" if v % 1 != 0 else f"{int(v)}%" for v in sorted_x])
    
    plt.ylim(-5, 105) 
    plt.yticks(range(0, 101, 10), [f"{v}%" for v in range(0, 101, 10)])
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
    
    plt.tight_layout()
    output_plot = f"punishment_{sim.punishment}_attack_{sim.attack_type}_sum_up_transaction_trend_{sim.algorithm}.png"
    plt.savefig(output_plot, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative plot successfully saved as: {output_plot}")

def save_summary_csv(sim):
    """
    At the end of the simulation, save a summary CSV file with the results.
    """

    total_nodes = len(sim.node_behaviors)
    counts = Counter(sim.node_behaviors.values())            # Quantity of honest and malicious nodes at the end of the simulation
    total_trades = sim.trusted_trades + sim.untrusted_trades
    
    avg_scams_final = (sum(sim.completed_scams_records) / len(sim.completed_scams_records)) if sim.completed_scams_records else 0
    success_rate_final = (total_trades / sim.total_attempts * 100) if sim.total_attempts > 0 else 0
    
    filename = f"{sim.algorithm}_PPC_{sim.ping_pong_choice}_Punish_{sim.punishment}_Pool_{sim.poolTrade}_Attack_{sim.attack_type}.csv"
    
    header = [
        # 1. Config parameters 
        'Algorithm', 'Trust Threshold', 'Default Rep', 'Pool Trade Size', 
        'Initial Nodes', 'Initial Connections', 'Malicious Rate',
        
        # 2. Performance
        'Final Success Rate (%)', 'Avg Scams/Isolated', 
        
        # 3. Network state
        'Total Nodes (End)', 'Honest Nodes', 'Malicious Nodes', 'Isolated Honest Nodes', 'Isolated Malicious Nodes', 
        
        # 4. Global transactions
        'Total Attempts', 'Successful Trades', 'Good Transactions', 'Malicious Transactions', 
        
        # 5. Node transaction
        'Honest Good Trades', 'Honest Bad Trades', 'Malicious Good Trades', 'Malicious Bad Trades', 
        'Mean Trades/Honest', 'Mean Trades/Malicious', 
        
        # 6. Time simulation
        'History'
    ]
    
    # Map simulation metrics to the respective CSV columns
    data_row = [
        # 1. Config parameters 
        sim.algorithm.upper(),
        sim.trust_threshold,
        sim.default_rep,
        sim.poolTrade,
        sim.n_nodes,
        sim.start_connection,
        sim.malicious_rate,
        
        # 2. Performance
        f"{success_rate_final:.2f}",
        f"{avg_scams_final:.2f}",
        
        # 3. Network state
        total_nodes,
        counts['honest'],
        counts['malicious'],
        sim.honest_nodes_isolation,
        sim.malicious_nodes_isolation,
        
        # 4. Global transactions
        sim.total_attempts,
        total_trades,
        sim.trusted_trades,
        sim.untrusted_trades,
        
        # 5. Node transaction
        sim.honest_good_trades,
        sim.honest_bad_trades,
        sim.malicious_good_trades,
        sim.malicious_bad_trades,
        f"{sim.mean_trades_honest[0]:.2f}" if sim.mean_trades_honest[1] > 0 else "N/A",
        f"{sim.mean_trades_malicious[0]:.2f}" if sim.mean_trades_malicious[1] > 0 else "N/A",
        
        # 6. Time simulation
        json.dumps(sim.history)  # Serialize the history dictionary into a CSV-compatible JSON string
    ]

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)

    print(f"Simulation results appended to: {filename}")

def plot_results_attack(folder_path="csv", output_dir="plots"):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in directory: '{folder_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    if not all_data:
        print("No valid data could be extracted from CSV files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    if "Malicious Rate" not in combined_df.columns:
        print("Error: 'Malicious Rate' column not found in CSV files.")
        return

    if "Algorithm" in combined_df.columns:
        combined_df["Algorithm"] = combined_df["Algorithm"].apply(
            lambda x: (
                "SHORTEST_PATH"
                if "dijkstra" in str(x).lower()
                else str(x).strip().upper()
            )
        )

    desired_order = ["MAX_FLOW", "PPR", "SHORTEST_PATH"]

    def get_order_rank(alg_name):
        alg_str = str(alg_name).upper()
        for idx, key in enumerate(desired_order):
            if key in alg_str:
                return idx
        return len(desired_order)

    grouped = combined_df.groupby("Malicious Rate")
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]

    for mal_rate, group in grouped:
        print(f"Generating plots for Malicious Rate = {mal_rate}...")

        group_sorted = group.copy()
        group_sorted["sort_key"] = group_sorted["Algorithm"].apply(
            get_order_rank
        )
        group_sorted = group_sorted.sort_values(by="sort_key").drop(
            columns=["sort_key"]
        )

        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax_leg = fig.add_subplot(gs[1, 1])

        fig.suptitle(
            f"Simulation Analysis - Malicious Rate: {mal_rate}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            alg = row.get("Algorithm", f"ALGORITHM_{idx+1}")

            history_data = row["History"]
            if isinstance(history_data, str):
                history = json.loads(history_data)
            else:
                history = history_data

            t = history["t"]

            h_trusted = history["trusted"]
            h_untrusted = history["untrusted"]
            totals = [
                (tr + u) if (tr + u) > 0 else 1
                for tr, u in zip(h_trusted, h_untrusted)
            ]
            trusted_pct = [(tr / tot * 100) for tr, tot in zip(h_trusted, totals)]

            ax1.plot(
                t,
                trusted_pct,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

            sr_percentage = [val * 100 for val in history["success_rate"]]
            ax2.plot(
                t,
                sr_percentage,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

            avg_scams = history["avg_scams_per_node"]
            ax3.plot(
                t,
                avg_scams,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

        ax1.set_title(
            "Cumulative Honest Transactions (%)", fontsize=11, fontweight="bold"
        )
        ax1.set_xlabel("Time (t)", fontsize=10)
        ax1.set_ylabel("Honest (%)", fontsize=10)
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.set_title(
            "Pathfinding Success Rate (%)", fontsize=11, fontweight="bold"
        )
        ax2.set_xlabel("Time (t)", fontsize=10)
        ax2.set_ylabel("% Success", fontsize=10)
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.6)

        ax3.set_title(
            "Avg Scams per Malicious Node before Isolation",
            fontsize=11,
            fontweight="bold",
        )
        ax3.set_xlabel("Time (t)", fontsize=10)
        ax3.set_ylabel("Average Scam Count", fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.6)

        ax_leg.axis("off")
        handles, labels = ax1.get_legend_handles_labels()
        ax_leg.legend(
            handles,
            labels,
            loc="center",
            fontsize=9,
            title="Algorithms",
            title_fontsize=10,
            frameon=True,
            handletextpad=0.5,
            labelspacing=0.4,
            borderpad=0.6,
        )

        plt.tight_layout()

        safe_rate_str = str(mal_rate).replace(".", "_")
        output_filename = os.path.join(
            output_dir, f"plot_malicious_rate_{safe_rate_str}.png"
        )

        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f" -> Saved plot: {output_filename}")



def plot_results_combined(folder_path="csv", output_dir="plots"):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in directory: '{folder_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    if not all_data:
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    if "Algorithm" in combined_df.columns:
        combined_df["Algorithm"] = combined_df["Algorithm"].apply(
            lambda x: (
                "SHORTEST_PATH"
                if "dijkstra" in str(x).lower()
                else str(x).strip().upper()
            )
        )

    desired_order = ["MAX_FLOW", "PPR", "SHORTEST_PATH"]

    def get_order_rank(alg_name):
        alg_str = str(alg_name).upper()
        for idx, key in enumerate(desired_order):
            if key in alg_str:
                return idx
        return len(desired_order)

    grouped = combined_df.groupby("Malicious Rate")
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]

    for mal_rate, group in grouped:
        print(f"Generating combined plot for Malicious Rate = {mal_rate}...")

        group_sorted = group.copy()
        group_sorted["sort_key"] = group_sorted["Algorithm"].apply(
            get_order_rank
        )
        group_sorted = group_sorted.sort_values(by="sort_key").drop(
            columns=["sort_key"]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        fig.suptitle(
            f"Simulation Analysis - Malicious Rate: {mal_rate}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            alg = row.get("Algorithm", f"ALGORITHM_{idx+1}")

            history_data = row["History"]
            history = (
                json.loads(history_data)
                if isinstance(history_data, str)
                else history_data
            )

            t = history["t"]
            h_trusted = history["trusted"]
            h_untrusted = history["untrusted"]
            totals = [
                (tr + u) if (tr + u) > 0 else 1
                for tr, u in zip(h_trusted, h_untrusted)
            ]
            trusted_pct = [
                (tr / tot * 100) for tr, tot in zip(h_trusted, totals)
            ]
            sr_percentage = [val * 100 for val in history["success_rate"]]

            ax1.plot(
                t,
                trusted_pct,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )
            ax2.plot(
                t,
                sr_percentage,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

        # Plot 1: Honest Transactions
        ax1.set_title(
            "Cumulative Honest Transactions (%)", fontsize=11, fontweight="bold"
        )
        ax1.set_xlabel("Time (t)", fontsize=10)
        ax1.set_ylabel("Honest (%)", fontsize=10)
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Legenda piccola posizionata dentro il primo grafico
        ax1.legend(
            fontsize=8,
            loc="lower right",
            frameon=True,
            framealpha=0.85,
            handlelength=1.5,
            borderpad=0.4,
            labelspacing=0.3,
        )

        # Plot 2: Pathfinding Success Rate
        ax2.set_title(
            "Pathfinding Success Rate (%)", fontsize=11, fontweight="bold"
        )
        ax2.set_xlabel("Time (t)", fontsize=10)
        ax2.set_ylabel("% Success", fontsize=10)
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

        safe_rate_str = str(mal_rate).replace(".", "_")
        output_filename = os.path.join(
            output_dir, f"plot_malicious_rate_{safe_rate_str}.png"
        )
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_results_separated_files(folder_path="csv", output_dir="plots"):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in directory: '{folder_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    if not all_data:
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    if "Algorithm" in combined_df.columns:
        combined_df["Algorithm"] = combined_df["Algorithm"].apply(
            lambda x: (
                "SHORTEST_PATH"
                if "dijkstra" in str(x).lower()
                else str(x).strip().upper()
            )
        )

    desired_order = ["MAX_FLOW", "PPR", "SHORTEST_PATH"]

    def get_order_rank(alg_name):
        alg_str = str(alg_name).upper()
        for idx, key in enumerate(desired_order):
            if key in alg_str:
                return idx
        return len(desired_order)

    grouped = combined_df.groupby("Malicious Rate")
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]

    for mal_rate, group in grouped:
        print(f"Generating separate PNGs for Malicious Rate = {mal_rate}...")

        group_sorted = group.copy()
        group_sorted["sort_key"] = group_sorted["Algorithm"].apply(
            get_order_rank
        )
        group_sorted = group_sorted.sort_values(by="sort_key").drop(
            columns=["sort_key"]
        )

        safe_rate_str = str(mal_rate).replace(".", "_")

        # ---------------- 1. Plot Honest Transactions ----------------
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))

        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            alg = row.get("Algorithm", f"ALGORITHM_{idx+1}")

            history_data = row["History"]
            history = (
                json.loads(history_data)
                if isinstance(history_data, str)
                else history_data
            )

            t = history["t"]
            h_trusted = history["trusted"]
            h_untrusted = history["untrusted"]
            totals = [
                (tr + u) if (tr + u) > 0 else 1
                for tr, u in zip(h_trusted, h_untrusted)
            ]
            trusted_pct = [
                (tr / tot * 100) for tr, tot in zip(h_trusted, totals)
            ]

            ax1.plot(
                t,
                trusted_pct,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

        ax1.set_title(
            f"Honest Transactions (%) - Malicious Rate: {mal_rate}",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("Time (t)", fontsize=10)
        ax1.set_ylabel("Honest (%)", fontsize=10)
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(
            fontsize=8,
            loc="lower right",
            frameon=True,
            framealpha=0.85,
            handlelength=1.5,
            borderpad=0.4,
            labelspacing=0.3,
        )

        plt.tight_layout()
        file_honest = os.path.join(
            output_dir, f"plot_honest_rate_{safe_rate_str}.png"
        )
        plt.savefig(file_honest, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # ---------------- 2. Plot Pathfinding Success Rate ----------------
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            alg = row.get("Algorithm", f"ALGORITHM_{idx+1}")

            history_data = row["History"]
            history = (
                json.loads(history_data)
                if isinstance(history_data, str)
                else history_data
            )

            t = history["t"]
            sr_percentage = [val * 100 for val in history["success_rate"]]

            ax2.plot(
                t,
                sr_percentage,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

        ax2.set_title(
            f"Pathfinding Success Rate (%) - Malicious Rate: {mal_rate}",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Time (t)", fontsize=10)
        ax2.set_ylabel("% Success", fontsize=10)
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(
            fontsize=8,
            loc="lower right",
            frameon=True,
            framealpha=0.85,
            handlelength=1.5,
            borderpad=0.4,
            labelspacing=0.3,
        )

        plt.tight_layout()
        file_success = os.path.join(
            output_dir, f"plot_success_rate_{safe_rate_str}.png"
        )
        plt.savefig(file_success, dpi=300, bbox_inches="tight")
        plt.close(fig2)


def plot_results_attack_only_2(folder_path="csv", output_dir="plots"):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in directory: '{folder_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    if not all_data:
        print("No valid data could be extracted from CSV files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    if "Malicious Rate" not in combined_df.columns:
        print("Error: 'Malicious Rate' column not found in CSV files.")
        return

    if "Algorithm" in combined_df.columns:
        combined_df["Algorithm"] = combined_df["Algorithm"].apply(
            lambda x: (
                "SHORTEST_PATH"
                if "dijkstra" in str(x).lower()
                else str(x).strip().upper()
            )
        )

    desired_order = ["MAX_FLOW", "PPR", "SHORTEST_PATH"]

    def get_order_rank(alg_name):
        alg_str = str(alg_name).upper()
        for idx, key in enumerate(desired_order):
            if key in alg_str:
                return idx
        return len(desired_order)

    grouped = combined_df.groupby("Malicious Rate")
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]

    for mal_rate, group in grouped:
        print(f"Generating plots for Malicious Rate = {mal_rate}...")

        group_sorted = group.copy()
        group_sorted["sort_key"] = group_sorted["Algorithm"].apply(
            get_order_rank
        )
        group_sorted = group_sorted.sort_values(by="sort_key").drop(
            columns=["sort_key"]
        )

        # 2 plot affiancati (1 riga, 2 colonne)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

        fig.suptitle(
            f"Simulation Analysis - Malicious Rate: {mal_rate}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        for idx, (_, row) in enumerate(group_sorted.iterrows()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            alg = row.get("Algorithm", f"ALGORITHM_{idx+1}")

            history_data = row["History"]
            if isinstance(history_data, str):
                history = json.loads(history_data)
            else:
                history = history_data

            t = history["t"]

            h_trusted = history["trusted"]
            h_untrusted = history["untrusted"]
            totals = [
                (tr + u) if (tr + u) > 0 else 1
                for tr, u in zip(h_trusted, h_untrusted)
            ]
            trusted_pct = [
                (tr / tot * 100) for tr, tot in zip(h_trusted, totals)
            ]

            ax1.plot(
                t,
                trusted_pct,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

            sr_percentage = [val * 100 for val in history["success_rate"]]
            ax2.plot(
                t,
                sr_percentage,
                label=alg,
                color=color,
                linewidth=2,
                linestyle=linestyle,
            )

        # 1. Cumulative Honest Transactions
        ax1.set_title(
            "Cumulative Honest Transactions (%)", fontsize=11, fontweight="bold"
        )
        ax1.set_xlabel("Time (t)", fontsize=10)
        ax1.set_ylabel("Honest (%)", fontsize=10)
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # 2. Pathfinding Success Rate
        ax2.set_title(
            "Pathfinding Success Rate (%)", fontsize=11, fontweight="bold"
        )
        ax2.set_xlabel("Time (t)", fontsize=10)
        ax2.set_ylabel("% Success", fontsize=10)
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # Estrazione degli elementi della legenda
        handles, labels = ax1.get_legend_handles_labels()

        # Legenda esterna in basso, al centro tra i 2 grafici
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(labels), 4),  # Distribuisce gli algoritmi su più colonne
            fontsize=9,
            title="Algorithms",
            title_fontsize=10,
            frameon=True,
            handletextpad=0.5,
            columnspacing=1.5,
        )

        # Spazio reserved per il titolo in alto (0.93) e la legenda in basso (0.12)
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])

        safe_rate_str = str(mal_rate).replace(".", "_")
        output_filename = os.path.join(
            output_dir, f"plot_malicious_rate_{safe_rate_str}.png"
        )

        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f" -> Saved plot: {output_filename}")


def save_roles_csv(sim):
    filename = f"{sim.algorithm}_Emb_{sim.embedded_type}_PPC_{sim.ping_pong_choice}_Punish_{sim.punishment}_Pool_{sim.poolTrade}_Attack_{sim.attack_type}_roles_outcomes.csv"
    
    total_trades = sim.honest_src_trades + sim.malicious_src_trades
    
    if total_trades > 0:
        pct_honest_source = (sim.honest_src_trades / total_trades) * 100
        pct_malicious_source = (sim.malicious_src_trades / total_trades) * 100
        pct_honest_destination = (sim.honest_dst_trades / total_trades) * 100
        pct_malicious_destination = (sim.malicious_dst_trades / total_trades) * 100
        
        pct_honest_good_trades = (sim.honest_good_trades / total_trades) * 100
        pct_honest_bad_trades = (sim.honest_bad_trades / total_trades) * 100
        pct_malicious_good_trades = (sim.malicious_good_trades / total_trades) * 100
        pct_malicious_bad_trades = (sim.malicious_bad_trades / total_trades) * 100
    else:
        pct_honest_source = 0.0
        pct_malicious_source = 0.0
        pct_honest_destination = 0.0
        pct_malicious_destination = 0.0
        pct_honest_good_trades = 0.0
        pct_honest_bad_trades = 0.0
        pct_malicious_good_trades = 0.0
        pct_malicious_bad_trades = 0.0

    header = [
        'Algorithm Name',
        'Ping Pong Choice',
        'Malicious Rate',
        'Percentage Honest Source',
        'Percentage Malicious Source',
        'Percentage Honest Destination',
        'Percentage Malicious Destination',
        'Percentage Honest Good Trades',
        'Percentage Honest Bad Trades',
        'Percentage Malicious Good Trades',
        'Percentage Malicious Bad Trades'
    ]
    
    data_row = [
        sim.algorithm.upper(),
        sim.ping_pong_choice,
        sim.malicious_rate,
        f"{pct_honest_source:.2f}",
        f"{pct_malicious_source:.2f}",
        f"{pct_honest_destination:.2f}",
        f"{pct_malicious_destination:.2f}",
        f"{pct_honest_good_trades:.2f}",
        f"{pct_honest_bad_trades:.2f}",
        f"{pct_malicious_good_trades:.2f}",
        f"{pct_malicious_bad_trades:.2f}"
    ]

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)

    print(f"Roles metrics appended to: {filename}")
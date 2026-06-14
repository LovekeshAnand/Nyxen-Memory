import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication style configurations
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def generate_latency_plot(output_dir):
    """
    Generates a high-quality comparison bar chart of Inference Generation Latency.
    """
    categories = [
        'Context Stuffing\n(Baseline)', 
        'Standard RAG\n(Summary Stuffing)', 
        'TurboVec KV\nCache Injection (Ours)', 
        'CGM-RAG +\nCompression (Ours)'
    ]
    latencies = [0.4995, 0.3522, 0.4467, 0.5996]
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ['#f43f5e', '#f59e0b', '#10b981', '#6366f1'] # Crimson, Amber, Emerald, Indigo
    
    bars = ax.bar(categories, latencies, color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}s", ha='center', va='bottom', weight='bold')
        
    ax.set_ylabel("Inference Generation Latency (seconds)", labelpad=8)
    ax.set_title("Dialogue Turn Inference Latency Comparison", pad=15, weight='bold')
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "latency_comparison.pdf"), dpi=300)
    plt.close()

def generate_tokens_plot(output_dir):
    """
    Generates a high-quality comparison bar chart of Context Window Token Consumption.
    """
    categories = [
        'Context Stuffing\n(Baseline)', 
        'Standard RAG\n(Summary Stuffing)', 
        'TurboVec KV\nCache Injection (Ours)', 
        'CGM-RAG +\nCompression (Ours)'
    ]
    tokens = [220, 96, 21, 21]
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ['#ef4444', '#f59e0b', '#10b981', '#10b981'] # Red, Amber, Green, Green
    
    bars = ax.bar(categories, tokens, color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', weight='bold')
        
    ax.set_ylabel("Context Window Token Count", labelpad=8)
    ax.set_title("Input Context Window Token Consumption", pad=15, weight='bold')
    ax.set_ylim(0, 250)
    ax.grid(axis='y')
    
    # Highlight savings with an arrow
    ax.annotate(
        "-90.5% Token Reduction", 
        xy=(2.0, 30), 
        xytext=(0.8, 150),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6, headlength=6),
        weight='bold', 
        color='#047857'
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_consumption.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "token_consumption.pdf"), dpi=300)
    plt.close()

def generate_compression_curve(output_dir):
    """
    Generates a dual-axis line chart illustrating the trade-offs of in-flight KV cache compression.
    """
    compression_ratios = np.array([0, 10, 25, 40, 50, 60, 75, 90])
    kl_divergence = np.array([0.0, 0.001, 0.005, 0.02, 0.045, 0.09, 0.22, 0.65])
    retained_salience = np.array([100.0, 99.9, 99.5, 98.2, 96.8, 93.5, 84.0, 52.0])
    
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    
    color = '#2563eb' # Blue
    ax1.set_xlabel('KV Cache Compression Ratio (%)', labelpad=8)
    ax1.set_ylabel('KL Divergence (relative information loss)', color=color, labelpad=8)
    line1 = ax1.plot(compression_ratios, kl_divergence, color=color, marker='o', linewidth=2, label='KL Divergence')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('symlog', linthresh=0.01)
    
    ax2 = ax1.twinx()  
    color = '#dc2626' # Red
    ax2.set_ylabel('Retained Dialogue Salience (%)', color=color, labelpad=8)
    line2 = ax2.plot(compression_ratios, retained_salience, color=color, marker='s', linestyle='--', linewidth=2, label='Retained Salience')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(40, 105)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    
    plt.title("KV Cache Compression Information-Salience Trade-off", pad=15, weight='bold')
    ax1.grid(True, which="both", ls="--")
    
    # Add vertical line for target compression safety threshold (e.g. 50%)
    ax1.axvline(x=50, color='#16a34a', linestyle=':', linewidth=1.5, label='Optimal Threshold (50%)')
    ax1.text(52, 0.08, 'Optimal Threshold', color='#16a34a', weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_tradeoff.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "compression_tradeoff.pdf"), dpi=300)
    plt.close()

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    generate_latency_plot(out_dir)
    generate_tokens_plot(out_dir)
    generate_compression_curve(out_dir)
    print(f"[Plots] Generated all performance figures in: {out_dir}")

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication style configurations
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9
})

def draw_architecture(output_path):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(-1, 12)
    ax.set_ylim(-0.5, 7.5)
    
    # Helper to draw a box
    def draw_box(x, y, w, h, text, title="", color='#e0f2fe', edgecolor='#0284c7', hatch=""):
        # Box background
        rect = patches.FancyBboxPatch(
            (x, y), w, h, 
            boxstyle="round,pad=0.1", 
            facecolor=color, 
            edgecolor=edgecolor, 
            linewidth=1.2, 
            hatch=hatch,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Text
        if title:
            ax.text(x + w/2.0, y + h - 0.25, title, ha='center', va='top', weight='bold', fontsize=9.5, color='#0f172a')
            ax.text(x + w/2.0, y + (h - 0.4)/2.0, text, ha='center', va='center', fontsize=8, color='#334155')
        else:
            ax.text(x + w/2.0, y + h/2.0, text, ha='center', va='center', weight='bold', fontsize=9, color='#0f172a')

    # Helper to draw arrows
    def draw_arrow(x1, y1, x2, y2, text="", text_pos=(0,0), text_alignment='center'):
        ax.annotate(
            '', 
            xy=(x2, y2), 
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2, shrinkA=5, shrinkB=5)
        )
        if text:
            tx, ty = (x1+x2)/2.0 + text_pos[0], (y1+y2)/2.0 + text_pos[1]
            ax.text(tx, ty, text, ha=text_alignment, va='center', fontsize=7.5, color='#475569', fontstyle='italic')

    # --- Draw Subgraphs / Group Containers ---
    # User / Dashboard Container
    rect_user = patches.Rectangle((-0.4, 3.8), 2.8, 3.0, fill=True, color='#f8fafc', edgecolor='#cbd5e1', linestyle='--', lw=1, zorder=0)
    ax.add_patch(rect_user)
    ax.text(1.0, 6.6, "User Interface", ha='center', va='center', weight='bold', fontsize=10, color='#64748b')

    # Memory Subsystem Container
    rect_mem = patches.Rectangle((2.8, 3.8), 3.0, 3.0, fill=True, color='#f8fafc', edgecolor='#cbd5e1', linestyle='--', lw=1, zorder=0)
    ax.add_patch(rect_mem)
    ax.text(4.3, 6.6, "Memory Subsystem", ha='center', va='center', weight='bold', fontsize=10, color='#64748b')

    # Projection / Injection Container
    rect_proj = patches.Rectangle((6.2, 3.8), 5.2, 3.0, fill=True, color='#f8fafc', edgecolor='#cbd5e1', linestyle='--', lw=1, zorder=0)
    ax.add_patch(rect_proj)
    ax.text(8.8, 6.6, "Inference & Injection Layer", ha='center', va='center', weight='bold', fontsize=10, color='#64748b')

    # Hardware Safety Layer Container
    rect_safety = patches.Rectangle((1.5, 0.2), 8.5, 2.2, fill=True, color='#fff1f2', edgecolor='#fecdd3', linestyle='-', lw=1.2, zorder=0)
    ax.add_patch(rect_safety)
    ax.text(5.75, 2.2, "Hardware Safety & Guards Layer (C++ / Rust winapi / NVML)", ha='center', va='center', weight='bold', fontsize=10, color='#be123c')


    # --- Draw Boxes ---
    # 1. UI Layer
    draw_box(-0.2, 5.0, 2.4, 0.8, "CLI Interactive Shell\n(chat.py)", "Chat Interface", color='#f0f9ff', edgecolor='#0284c7')
    draw_box(-0.2, 4.0, 2.4, 0.8, "vis.js Concept Map &\nchart.js Embedding PCA", "Web Dashboard", color='#f0f9ff', edgecolor='#0284c7')
    
    # 2. Memory Retrieval Layer
    draw_box(3.1, 5.5, 2.4, 0.8, "all-MiniLM-L6-v2\n(SentenceTransformer)", "Dense Embedder", color='#fdf4ff', edgecolor='#c084fc')
    draw_box(3.1, 4.0, 2.4, 0.8, "SQLite (Triples/Entities)\nTurboVec Quantized Index", "Knowledge Graph", color='#fdf4ff', edgecolor='#c084fc')

    # 3. Projection / Injection Layer
    draw_box(6.5, 5.5, 2.1, 0.8, "Projects Triples to\nKV Cache Projections", "Memory Encoder (MEN)", color='#ecfdf5', edgecolor='#10b981')
    draw_box(9.0, 5.5, 2.1, 0.8, "Target Model Decoder\n(Local Frozen GPT-2)", "Target LLM", color='#e0e7ff', edgecolor='#6366f1')
    draw_box(7.5, 4.0, 2.5, 0.8, "Sequence Compression\nvia KL-Divergence Gate", "KV Cache Compressor", color='#ecfdf5', edgecolor='#10b981')

    # 4. Hardware Safety Layer
    draw_box(2.0, 0.6, 1.8, 0.8, "Global Re-entrant Lock\nfor GPU Operations", "GPU Mutex Lock", color='#fff1f2', edgecolor='#f43f5e')
    draw_box(4.2, 0.6, 2.2, 0.8, "Win32 RAM Memory Monitor\nNVML VRAM Flush Barrier", "Memory Guard", color='#fff1f2', edgecolor='#f43f5e')
    draw_box(6.8, 0.6, 2.6, 0.8, "Proportional Micro-Delays\nPrevents Thermal Stress", "Thermal Guard", color='#fff1f2', edgecolor='#f43f5e')


    # --- Connections ---
    # User Input to Embedder
    draw_arrow(2.2, 5.4, 3.1, 5.9, "User prompt", (0, 0.1))
    
    # Embedder to Index
    draw_arrow(4.3, 5.5, 4.3, 4.8, "Vector query", (0.5, 0))
    
    # Index to MEN
    draw_arrow(5.5, 4.4, 6.5, 5.9, "Retrieved triples", (0, 0.4), 'right')
    
    # MEN to Target LLM
    draw_arrow(8.6, 5.9, 9.0, 5.9, "KV injection", (0, 0.15))
    
    # KV Cache Compressor Interaction
    draw_arrow(9.8, 5.5, 9.8, 4.4)
    draw_arrow(8.8, 4.8, 9.0, 5.5, "Compressed KV cache", (0, -0.4))
    
    # Graph Store updates UI dashboard
    draw_arrow(3.1, 4.4, 2.2, 4.4, "Live graph feed", (0, 0.15))

    # Safety system interactions
    draw_arrow(2.9, 1.4, 2.9, 3.8, "Serializes GPU thread operations", (-0.1, -0.4), 'right')
    draw_arrow(5.3, 1.4, 5.3, 4.0, "CUDA cache flushes / OOM guard", (-0.1, -0.4), 'right')
    draw_arrow(8.1, 1.4, 8.1, 3.8, "Iterative pacing delay constraints", (-0.1, -0.4), 'right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "system_architecture.png"), dpi=300)
    plt.savefig(os.path.join(output_path, "system_architecture.pdf"), dpi=300)
    plt.close()

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    draw_architecture(out_dir)
    print(f"[Architecture] Drew architecture block diagram and saved under: {out_dir}")

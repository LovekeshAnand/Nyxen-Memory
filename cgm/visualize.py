import os
import sys

# Support running directly from inside the cgm directory or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import networkx as nx
from cgm.schema import SQLiteGraphStore

def generate_graph_visualization(db_path: str = "data/cgm_memory.db", output_image: str = "data/memory_graph.png"):
    """
    Queries the SQLite Graph Store, builds a directed networkx graph of the
    retrieved semantic triples, and renders a stunning dark-themed visualization.
    """
    print("=" * 60)
    print("      CONVERSATIONAL GRAPH MEMORY (CGM) - GRAPH VISUALIZATION")
    print("=" * 60)
    
    if not os.path.exists(db_path):
        print(f"[Visualizer] Database not found at '{db_path}'. Please run the demo first to seed it.")
        return
        
    print(f"[Visualizer] Reading graph database from: {db_path}...")
    store = SQLiteGraphStore(db_path)
    
    # Retrieve test conversation graph data
    conversation_id = "test_conversation_99"
    graph_data = store.get_conversation_graph(conversation_id)
    triples = graph_data["triples"]
    
    if not triples:
        print("[Visualizer] No triples found in database. Run 'python run_demo.py' first.")
        return
        
    print(f"[Visualizer] Loaded {len(triples)} memory triples. Rending graph diagram...")
    
    # Initialize Directed Graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    edge_labels = {}
    for trip in triples:
        subj = trip["subject"]
        pred = trip["predicate"]
        obj = trip["object"]
        
        G.add_edge(subj, obj)
        edge_labels[(subj, obj)] = pred
        
    # Set dark-themed figure size and styling
    plt.figure(figsize=(12, 10), facecolor='#111827') # Sleek Tailored Dark Gray
    ax = plt.gca()
    ax.set_facecolor('#111827')
    
    # Apply spring layout with adjusted optimal distance
    pos = nx.spring_layout(G, k=1.8, seed=42)
    
    # Draw Nodes (Glassmorphism look with bright cyan gradient)
    nx.draw_networkx_nodes(
        G, pos,
        node_color='#06B6D4', # Neon Cyan
        node_size=2800,
        alpha=0.85,
        edgecolors='#22D3EE', # Bright Turquoise Border
        linewidths=2
    )
    
    # Draw Edges (Translucent light gray arrows)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#9CA3AF',
        width=1.5,
        arrowsize=20,
        alpha=0.6,
        arrowstyle='-|>'
    )
    
    # Draw Labels (Modern Typography over dark background)
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color='#F3F4F6', # Soft White
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # Draw Edge Labels (Relations/Predicates in neon violet/purple)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color='#C084FC', # Light Violet
        font_weight='bold',
        font_family='sans-serif',
        bbox=dict(facecolor='#1F2937', edgecolor='none', alpha=0.9, boxstyle='round,pad=0.3')
    )
    
    plt.title(
        f"Conversational Graph Memory (CGM)\nActive Concept map for '{conversation_id}'",
        fontsize=16,
        color='#22D3EE',
        weight='bold',
        pad=20
    )
    
    # Remove axis borders
    plt.axis('off')
    
    # Ensure parent output dir exists
    output_dir = os.path.dirname(output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_image, facecolor='#111827', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[Visualizer] SUCCESS! Generated conversational graph visualization saved to: {output_image}")
    print("=" * 60)

if __name__ == "__main__":
    generate_graph_visualization()

import os
import sys
import json
import time
import socket
import webbrowser
import threading

# Configure stdout and stderr to handle UTF-8 encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from http.server import SimpleHTTPRequestHandler, HTTPServer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional

from cgm.database.schema import SQLiteGraphStore

# Global circular buffer for pipeline activities (real-time visualizer streaming)
PIPELINE_LOGS = []

def log_pipeline_event(event_type: str, details: dict):
    """
    Appends a new event log to the circular log buffer.
    """
    global PIPELINE_LOGS
    PIPELINE_LOGS.append({
        "timestamp": time.time(),
        "type": event_type,
        "details": details
    })
    # Keep buffer capped at 100 entries
    if len(PIPELINE_LOGS) > 100:
        PIPELINE_LOGS.pop(0)

# Seed an initial log entry on module import
log_pipeline_event("info", {"message": "Pipeline event logging subsystem initialized."})


def generate_graph_visualization(conversation_id: str = "test_conversation_99", db_path: str = "data/cgm_memory.db", output_image: str = "data/memory_graph.png"):
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
    
    # Retrieve conversation graph data
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


def generate_embedding_clusters(db_path: str = "data/cgm_memory.db", output_image: str = "data/embedding_clusters.png", conversation_id: str = "test_conversation_99"):
    """
    Queries turn embeddings from SQLite, projects them to 2D using PCA,
    and draws a beautiful sequential trajectory scatter plot of the conversations.
    """
    if not os.path.exists(db_path):
        print(f"[Visualizer] Database not found at '{db_path}'. Skipping cluster generation.")
        return
        
    store = SQLiteGraphStore(db_path)
    with store._lock:
        conn = store._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.turn_id, t.summary, e.data, e.dim 
            FROM turns t JOIN turn_embeddings e 
            ON t.conversation_id = e.conversation_id AND t.turn_id = e.turn_id
            WHERE t.conversation_id = ? ORDER BY t.turn_id ASC
        """, (conversation_id,))
        rows = cursor.fetchall()
        conn.close()
        
    if len(rows) < 2:
        print("[Visualizer] Too few embeddings to draw a 2D cluster map. Seed at least 2 turns first.")
        return
        
    embeddings = []
    for r in rows:
        arr = np.frombuffer(r["data"], dtype=np.float32).reshape((r["dim"],))
        embeddings.append(arr)
    embeddings = np.stack(embeddings)
    
    # Project to 2D using PCA
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
    except Exception as e:
        print(f"[Visualizer] PCA calculation failed: {e}")
        return
        
    plt.figure(figsize=(10, 8), facecolor='#0b0f19')
    ax = plt.gca()
    ax.set_facecolor('#0b0f19')
    
    # Scatter points color-coded by sequence timeline
    colors = plt.cm.cool(np.linspace(0, 1, len(rows)))
    
    # Draw trajectory connections
    for i in range(len(rows) - 1):
        plt.annotate(
            "", 
            xy=(coords[i+1, 0], coords[i+1, 1]), 
            xytext=(coords[i, 0], coords[i, 1]),
            arrowprops=dict(arrowstyle="->", color="rgba(156, 163, 175, 0.4)", lw=1.5)
        )
        
    # Plot points
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(rows)), cmap='cool', s=250, edgecolors='#22D3EE', linewidths=1.5, zorder=5)
    
    # Label each turn point
    for idx, r in enumerate(rows):
        summary_trunc = r["summary"][:20] + "..." if len(r["summary"]) > 20 else r["summary"]
        plt.text(
            coords[idx, 0] + 0.01, 
            coords[idx, 1] + 0.01, 
            f"T{r['turn_id']}: {summary_trunc}", 
            color='#F3F4F6', 
            fontsize=8, 
            fontweight='bold',
            bbox=dict(facecolor='#1F2937', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2')
        )
        
    plt.title("Conversational Trajectory in Latent Space", color='#22D3EE', fontsize=14, weight='bold', pad=15)
    plt.xlabel("Principal Component 1", color='#9CA3AF')
    plt.ylabel("Principal Component 2", color='#9CA3AF')
    ax.tick_params(colors='#9CA3AF')
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, color='rgba(55, 65, 81, 0.3)')
    
    output_dir = os.path.dirname(output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_image, facecolor='#0b0f19', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Visualizer] SUCCESS! Generated latent cluster visualization saved to: {output_image}")


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PIPELINE CONTROL CENTER / DASHBOARD (GLASSMORPHISM THEME)
# ═══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversational Graph Memory (CGM) - Real-time Pipeline Dashboard</title>
    <!-- vis.js CDN for concept map network drawing -->
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <!-- Chart.js CDN for PCA coordinates plotting -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        body {
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at center, #0c0f24 0%, #060814 100%);
            color: #f3f4f6;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 25px;
            background: rgba(15, 23, 42, 0.45);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(6, 182, 212, 0.25);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            z-index: 10;
        }
        
        .header-brand h1 {
            margin: 0;
            font-size: 1.25rem;
            font-weight: 800;
            background: linear-gradient(90deg, #22d3ee 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .header-brand p {
            margin: 2px 0 0 0;
            font-size: 0.75rem;
            color: #9ca3af;
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            padding: 5px 12px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 9999px;
            color: #34d399;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.8s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .dashboard-grid {
            flex: 1;
            display: grid;
            grid-template-columns: 45% 55%;
            padding: 15px;
            gap: 15px;
            height: calc(100vh - 65px);
            box-sizing: border-box;
        }

        .col-left, .col-right {
            display: flex;
            flex-direction: column;
            gap: 15px;
            height: 100%;
            overflow: hidden;
        }

        .card {
            background: rgba(15, 23, 42, 0.4);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            position: relative;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.35), transparent);
        }
        
        .card.card-purple::before {
            background: linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.35), transparent);
        }

        .card-header {
            padding: 10px 15px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.85rem;
            font-weight: 700;
            color: #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .card-header span {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-body {
            flex: 1;
            padding: 12px;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* 1. Vis.js Concept Map */
        #concept-graph-container {
            width: 100%;
            height: 100%;
            background: rgba(3, 7, 18, 0.2);
            border-radius: 8px;
        }

        /* 2. Chart.js Scatter Plot */
        .chart-wrapper {
            flex: 1;
            position: relative;
            min-height: 0;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        #trajectory-chart {
            width: 100% !important;
            height: 100% !important;
        }

        /* 3. Turns Database Inspector */
        .table-wrapper {
            flex: 1;
            overflow-y: auto;
            border-radius: 8px;
            background: rgba(3, 7, 18, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
            text-align: left;
        }

        th, td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        th {
            background: rgba(15, 23, 42, 0.6);
            color: #9ca3af;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        
        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .turn-badge {
            background: rgba(168, 85, 247, 0.25);
            border: 1px solid rgba(168, 85, 247, 0.5);
            color: #c084fc;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
            font-family: monospace;
        }
        
        /* Barcode Heatmap Visualizer */
        .barcode-container {
            display: flex;
            gap: 1.5px;
            align-items: center;
            background: #030712;
            padding: 3px;
            border-radius: 3px;
            width: fit-content;
        }
        
        .barcode-cell {
            width: 5px;
            height: 14px;
            border-radius: 0.5px;
        }

        /* 4. Console Logs */
        .console-container {
            flex: 1;
            background: #030712;
            border-radius: 8px;
            padding: 10px;
            font-family: 'Fira Code', 'Courier New', Courier, monospace;
            font-size: 0.72rem;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.03);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        
        .log-line {
            display: flex;
            gap: 10px;
            line-height: 1.4;
        }
        
        .log-time {
            color: #4b5563;
            flex-shrink: 0;
        }
        
        .log-tag {
            font-weight: 700;
            flex-shrink: 0;
            padding: 1px 4px;
            border-radius: 3px;
            text-transform: uppercase;
            font-size: 0.65rem;
        }
        
        .log-tag.info { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .log-tag.store { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .log-tag.retrieve { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .log-tag.generate { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }
        .log-tag.compress { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
        
        .log-text {
            color: #d1d5db;
            word-break: break-all;
        }

        /* 5. Gauges */
        .system-gauges {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .gauge-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .gauge-info {
            display: flex;
            flex-direction: column;
        }
        
        .gauge-label {
            font-size: 0.65rem;
            color: #9ca3af;
            font-weight: 700;
            text-transform: uppercase;
        }
        
        .gauge-val {
            font-size: 0.85rem;
            font-weight: 800;
            color: #22d3ee;
            font-family: monospace;
        }
        
        .gauge-bar-outer {
            width: 80px;
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 2px;
        }
        
        .gauge-bar-inner {
            height: 100%;
            background: linear-gradient(90deg, #06b6d4, #a855f7);
            width: 0%;
            transition: width 0.5s ease-in-out;
        }
        
    </style>
</head>
<body>
    <header>
        <div class="header-brand">
            <h1>🌌 Conversational Graph Memory</h1>
            <p>Active RAG Pipeline & KV-Injection Dashboard</p>
        </div>
        
        <div class="header-status">
            <div class="system-gauges">
                <div class="gauge-item">
                    <div class="gauge-info">
                        <span class="gauge-label">System RAM</span>
                        <span id="ram-text" class="gauge-val">0.0 GB / 0.0 GB</span>
                        <div class="gauge-bar-outer"><div id="ram-bar" class="gauge-bar-inner"></div></div>
                    </div>
                </div>
                <div class="gauge-item">
                    <div class="gauge-info">
                        <span class="gauge-label">GPU VRAM</span>
                        <span id="vram-text" class="gauge-val">0.0 GB / 0.0 GB</span>
                        <div class="gauge-bar-outer"><div id="vram-bar" class="gauge-bar-inner"></div></div>
                    </div>
                </div>
            </div>
            
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Active GPU Session</span>
            </div>
        </div>
    </header>

    <div class="dashboard-grid">
        <!-- LEFT COLUMN: Graph & Live Terminal Logs -->
        <div class="col-left">
            <!-- Concept Map Graph -->
            <div class="card" style="flex: 6.5;">
                <div class="card-header">
                    <span>🧬 Symbolic Memory Concept Graph</span>
                    <span style="font-size:0.7rem; color:#06b6d4;">实时实体 & 关系网络 (VisJS)</span>
                </div>
                <div class="card-body">
                    <div id="concept-graph-container"></div>
                </div>
            </div>
            
            <!-- Terminal Live Logs -->
            <div class="card" style="flex: 3.5;">
                <div class="card-header">
                    <span>💻 Pipeline Activity Console Stream</span>
                    <span style="font-size:0.7rem; color:#34d399;">Real-time Pipeline Logs</span>
                </div>
                <div class="card-body">
                    <div id="console-stream" class="console-container">
                        <!-- Filled by JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT COLUMN: Embeddings Trajectory & Database Turn Inspector -->
        <div class="col-right">
            <!-- Latent Space Trajectory Graph -->
            <div class="card card-purple" style="flex: 5;">
                <div class="card-header">
                    <span>📈 Latent Space Conversational Trajectory</span>
                    <span style="font-size:0.7rem; color:#a855f7;">PCA 2D Projection of Turn Embeddings (Chart.js)</span>
                </div>
                <div class="card-body">
                    <div class="chart-wrapper">
                        <canvas id="trajectory-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Database HMO Inspector -->
            <div class="card card-purple" style="flex: 5;">
                <div class="card-header">
                    <span>🗄️ Turn Database Inspector (HMO Viewer)</span>
                    <span style="font-size:0.7rem; color:#9ca3af;">SQLite Turns & Embedding Barcode Heatmaps</span>
                </div>
                <div class="card-body">
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th style="width: 80px;">Turn ID</th>
                                    <th style="width: 150px;">Text Snippets</th>
                                    <th>Embedding Heatmap (First 32 Dimensions)</th>
                                </tr>
                            </thead>
                            <tbody id="turns-table-body">
                                <!-- Filled by JS -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        // 1. Vis.js Graph Init
        const graphContainer = document.getElementById('concept-graph-container');
        const nodes = new vis.DataSet();
        const edges = new vis.DataSet();
        const networkData = { nodes: nodes, edges: edges };
        
        const networkOptions = {
            nodes: {
                shape: 'dot',
                size: 20,
                font: {
                    color: '#f3f4f6',
                    size: 11,
                    face: 'Inter, sans-serif',
                    strokeWidth: 2,
                    strokeColor: '#0b0f19'
                },
                color: {
                    background: '#0891b2',
                    border: '#22d3ee',
                    highlight: { background: '#06b6d4', border: '#67e8f9' }
                },
                borderWidth: 2,
                shadow: { enabled: true, color: 'rgba(6, 182, 212, 0.3)', size: 10 }
            },
            edges: {
                width: 1.5,
                color: { color: 'rgba(156, 163, 175, 0.45)', highlight: '#c084fc' },
                font: { color: '#c084fc', size: 9, face: 'Inter, sans-serif', strokeWidth: 2, strokeColor: '#0b0f19' },
                arrows: { to: { enabled: true, scaleFactor: 0.6 } },
                smooth: { enabled: true, type: 'cubicBezier', roundness: 0.35 }
            },
            physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: { gravitationalConstant: -50, springLength: 80, springConstant: 0.05, damping: 0.4 },
                stabilization: { enabled: true, iterations: 100 }
            }
        };
        const network = new vis.Network(graphContainer, networkData, networkOptions);

        // 2. Chart.js Scatter Trajectory Init
        const ctx = document.getElementById('trajectory-chart').getContext('2d');
        const trajectoryChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Turn Trajectory',
                    data: [],
                    showLine: true,
                    borderColor: 'rgba(168, 85, 247, 0.5)',
                    borderWidth: 2,
                    backgroundColor: [],
                    pointBorderColor: '#22d3ee',
                    pointBorderWidth: 1.5,
                    pointRadius: 8,
                    pointHoverRadius: 11,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const pt = context.raw;
                                return `Turn ${pt.turn_id}: ${pt.summary}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Principal Component 1', color: '#9ca3af' },
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#9ca3af' }
                    },
                    y: {
                        title: { display: true, text: 'Principal Component 2', color: '#9ca3af' },
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#9ca3af' }
                    }
                }
            }
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // DATA FETCHING & UI UPDATE LOOPS
        // ═══════════════════════════════════════════════════════════════════════════════
        
        let lastGraphHash = '';
        let lastTurnsHash = '';
        let lastLogsLength = 0;

        // Color mapper helper for embedding values: maps [-1.0, 1.0] to Blue -> White -> Red
        function valueToColor(val) {
            // Clamp val
            val = Math.max(-0.5, Math.min(0.5, val)) * 2; // amplify for color range
            if (val < 0) {
                // Negative: Blue scale
                const b = Math.floor(255);
                const r = Math.floor(255 * (1 + val));
                const g = Math.floor(255 * (1 + val));
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                // Positive: Red scale
                const r = Math.floor(255);
                const g = Math.floor(255 * (1 - val));
                const b = Math.floor(255 * (1 - val));
                return `rgb(${r}, ${g}, ${b})`;
            }
        }

        async function pollData() {
            try {
                // 1. System gauges
                const sysResponse = await fetch('/api/system');
                const sys = await sysResponse.json();
                if (!sys.error) {
                    document.getElementById('ram-text').innerText = `${sys.used_ram_gb} GB / ${sys.total_ram_gb} GB`;
                    document.getElementById('ram-bar').style.width = `${sys.ram_pct}%`;
                    document.getElementById('vram-text').innerText = `${sys.used_vram_gb} GB / ${sys.total_vram_gb} GB`;
                    document.getElementById('vram-bar').style.width = `${sys.vram_pct}%`;
                }

                // 2. Concept map graph
                const graphResponse = await fetch('/api/graph');
                const graph = await graphResponse.json();
                const currentGraphHash = JSON.stringify(graph);
                if (currentGraphHash !== lastGraphHash) {
                    lastGraphHash = currentGraphHash;
                    
                    const activeNodeIds = new Set();
                    graph.nodes.forEach(node => {
                        activeNodeIds.add(node.id);
                        if (nodes.get(node.id)) {
                            nodes.update(node);
                        } else {
                            nodes.add(node);
                        }
                    });
                    nodes.getIds().forEach(id => {
                        if (!activeNodeIds.has(id)) nodes.remove(id);
                    });

                    const activeEdgeIds = new Set();
                    graph.edges.forEach(edge => {
                        const edgeId = edge.from + '-' + edge.to + '-' + edge.label;
                        activeEdgeIds.add(edgeId);
                        edge.id = edgeId;
                        if (edges.get(edgeId)) {
                            edges.update(edge);
                        } else {
                            edges.add(edge);
                        }
                    });
                    edges.getIds().forEach(id => {
                        if (!activeEdgeIds.has(id)) edges.remove(id);
                    });
                }

                // 3. Turns table & embedding heatmap barcodes
                const turnsResponse = await fetch('/api/turns');
                const turns = await turnsResponse.json();
                const currentTurnsHash = JSON.stringify(turns);
                if (currentTurnsHash !== lastTurnsHash) {
                    lastTurnsHash = currentTurnsHash;
                    
                    const tbody = document.getElementById('turns-table-body');
                    tbody.innerHTML = '';
                    
                    turns.forEach(t => {
                        const tr = document.createElement('tr');
                        
                        // ID cell
                        const tdId = document.createElement('td');
                        tdId.innerHTML = `<span class="turn-badge">Turn ${t.turn_id}</span>`;
                        tr.appendChild(tdId);
                        
                        // Text cell
                        const tdText = document.createElement('td');
                        const sum_text = t.summary.length > 50 ? t.summary.substring(0, 50) + '...' : t.summary;
                        tdText.innerHTML = `<strong>Summary:</strong> ${sum_text}<br><span style="color:#9ca3af; font-size:0.7rem;">${t.user_text ? t.user_text.substring(0, 40) + '...' : ''}</span>`;
                        tr.appendChild(tdText);
                        
                        // Barcode heatmap cell
                        const tdBarcode = document.createElement('td');
                        const barcodeDiv = document.createElement('div');
                        barcodeDiv.className = 'barcode-container';
                        
                        if (t.embedding_slice && t.embedding_slice.length > 0) {
                            t.embedding_slice.forEach(val => {
                                const cell = document.createElement('div');
                                cell.className = 'barcode-cell';
                                cell.style.backgroundColor = valueToColor(val);
                                cell.title = `Dim value: ${val.toFixed(4)}`;
                                barcodeDiv.appendChild(cell);
                            });
                        } else {
                            barcodeDiv.innerHTML = '<span style="color:#4b5563; font-size:0.7rem; font-style:italic;">No embedding stored</span>';
                        }
                        tdBarcode.appendChild(barcodeDiv);
                        tr.appendChild(tdBarcode);
                        
                        tbody.appendChild(tr);
                    });
                    // Auto-scroll inspector to the bottom to track new turns
                    const inspectorBody = tbody.parentElement.parentElement;
                    inspectorBody.scrollTop = inspectorBody.scrollHeight;
                }

                // 4. Latent Space scatter plot trajectory
                const trajResponse = await fetch('/api/embeddings');
                const pts = await trajResponse.json();
                
                // Color mapper for trajectory coordinates: older turns are blue, newer are magenta
                const ptColors = pts.map((pt, idx) => {
                    const pct = idx / (pts.length - 1 || 1);
                    return `rgb(${Math.floor(168 + pct * 68)}, ${Math.floor(85 * (1 - pct))}, ${Math.floor(247 + pct * 8)})`;
                });
                
                trajectoryChart.data.datasets[0].data = pts;
                trajectoryChart.data.datasets[0].backgroundColor = ptColors;
                trajectoryChart.update();

                // 5. Console terminal logs stream
                const logsResponse = await fetch('/api/logs');
                const logs = await logsResponse.json();
                if (logs.length > lastLogsLength) {
                    const consoleContainer = document.getElementById('console-stream');
                    
                    // Slice new logs
                    const newLogs = logs.slice(lastLogsLength);
                    lastLogsLength = logs.length;
                    
                    newLogs.forEach(log => {
                        const line = document.createElement('div');
                        line.className = 'log-line';
                        
                        const date = new Date(log.timestamp * 1000);
                        const timeStr = date.toTimeString().split(' ')[0];
                        
                        // Determine event display message
                        let detailsMsg = log.details.message || JSON.stringify(log.details);
                        if (log.type === 'retrieve') {
                            detailsMsg = `RAG Search for: "${log.details.query}" | Matches: ` + 
                                log.details.results.map(r => `Turn ${r.turn_id} (Score: ${r.score.toFixed(3)})`).join(', ');
                        } else if (log.type === 'compress') {
                            detailsMsg = log.details.message;
                        }
                        
                        line.innerHTML = `
                            <span class="log-time">${timeStr}</span>
                            <span class="log-tag ${log.type}">${log.type}</span>
                            <span class="log-text">${detailsMsg}</span>
                        `;
                        consoleContainer.appendChild(line);
                    });
                    
                    // Scroll console to bottom
                    consoleContainer.scrollTop = consoleContainer.scrollHeight;
                }

            } catch (err) {
                console.error("Dashboard failed to poll updates", err);
            }
        }

        // Poll immediately and start interval
        pollData();
        setInterval(pollData, 1000);
    </script>
</body>
</html>
"""

def get_live_graph_json(conversation_id: str, db_path: str = "data/cgm_memory.db") -> str:
    """
    Queries SQLite Graph Store and maps relational triples into D3/VisJS-compatible nodes/edges.
    """
    store = SQLiteGraphStore(db_path)
    graph_data = store.get_conversation_graph(conversation_id)
    triples = graph_data["triples"]
    entities = graph_data["entities"]
    
    nodes = []
    seen_nodes = set()
    
    # Pre-populate named entities
    for ent in entities:
        name = ent["name"]
        nodes.append({
            "id": name,
            "label": name,
            "title": f"Type: {ent['type']}\\nDescription: {ent['description']}"
        })
        seen_nodes.add(name)
        
    # Ensure any subjects/objects in triples not in entities are added
    for trip in triples:
        for node in [trip["subject"], trip["object"]]:
            if node not in seen_nodes:
                nodes.append({
                    "id": node,
                    "label": node,
                    "title": f"Concept: {node}"
                })
                seen_nodes.add(node)
                
    # Build edges
    edges = []
    for trip in triples:
        edges.append({
            "from": trip["subject"],
            "to": trip["object"],
            "label": trip["predicate"]
        })
        
    return json.dumps({"nodes": nodes, "edges": edges})


def get_turns_with_embeddings(conversation_id: str, db_path: str) -> str:
    """
    Retrieves all turn rows for a conversation and appends sliced embedding vector elements.
    """
    store = SQLiteGraphStore(db_path)
    with store._lock:
        conn = store._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT turn_id, timestamp, summary, user_text, assistant_text 
                FROM turns WHERE conversation_id = ? ORDER BY turn_id ASC
            """, (conversation_id,))
            turns = [dict(row) for row in cursor.fetchall()]
            
            for t in turns:
                cursor.execute("""
                    SELECT data, dim FROM turn_embeddings
                    WHERE conversation_id = ? AND turn_id = ?
                """, (conversation_id, t["turn_id"]))
                row = cursor.fetchone()
                if row is not None:
                    arr = np.frombuffer(row["data"], dtype=np.float32)
                    t["embedding_slice"] = arr[:32].tolist()
                    t["embedding_dim"] = int(row["dim"])
                else:
                    t["embedding_slice"] = []
                    t["embedding_dim"] = 0
            return json.dumps(turns)
        except Exception as e:
            return json.dumps([{"error": str(e)}])
        finally:
            conn.close()


def get_2d_embeddings(conversation_id: str, db_path: str) -> str:
    """
    Fetches all turn embeddings and runs PCA on the fly to return 2D coordinates.
    """
    store = SQLiteGraphStore(db_path)
    with store._lock:
        conn = store._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT t.turn_id, t.summary, e.data, e.dim 
                FROM turns t JOIN turn_embeddings e 
                ON t.conversation_id = e.conversation_id AND t.turn_id = e.turn_id
                WHERE t.conversation_id = ? ORDER BY t.turn_id ASC
            """, (conversation_id,))
            rows = cursor.fetchall()
            
            if len(rows) < 2:
                points = []
                for i, r in enumerate(rows):
                    points.append({
                        "turn_id": r["turn_id"],
                        "summary": r["summary"],
                        "x": float(i * 1.5),
                        "y": 0.0
                    })
                return json.dumps(points)
                
            embeddings = []
            for r in rows:
                arr = np.frombuffer(r["data"], dtype=np.float32).reshape((r["dim"],))
                embeddings.append(arr)
            embeddings = np.stack(embeddings)
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            
            points = []
            for idx, r in enumerate(rows):
                points.append({
                    "turn_id": r["turn_id"],
                    "summary": r["summary"],
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1])
                })
            return json.dumps(points)
        except Exception as e:
            # Fallback coordinates on failure
            points = []
            for idx, r in enumerate(rows):
                points.append({
                    "turn_id": r["turn_id"],
                    "summary": r["summary"],
                    "x": float(idx),
                    "y": float(idx * 0.4)
                })
            return json.dumps(points)
        finally:
            conn.close()


def get_system_status() -> str:
    """
    Retrieves system RAM and GPU VRAM utilizing our dynamic safety library wrapper.
    """
    try:
        from cgm.safety.safety import GPUMemoryGuard
        guard = GPUMemoryGuard()
        free_ram, total_ram = guard.check_system_ram()
        free_vram, total_vram = guard.check_vram()
        
        used_ram = total_ram - free_ram
        used_vram = total_vram - free_vram
        
        ram_pct = (used_ram / total_ram * 100) if total_ram > 0 else 0
        vram_pct = (used_vram / total_vram * 100) if total_vram > 0 else 0
        
        return json.dumps({
            "free_ram_gb": round(free_ram / (1024**3), 2),
            "total_ram_gb": round(total_ram / (1024**3), 2),
            "used_ram_gb": round(used_ram / (1024**3), 2),
            "ram_pct": round(ram_pct, 1),
            
            "free_vram_gb": round(free_vram / (1024**3), 2),
            "total_vram_gb": round(total_vram / (1024**3), 2),
            "used_vram_gb": round(used_vram / (1024**3), 2),
            "vram_pct": round(vram_pct, 1)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_pipeline_logs() -> str:
    """
    Returns circular log buffer elements.
    """
    global PIPELINE_LOGS
    return json.dumps(PIPELINE_LOGS)


def find_available_port(start_port=8050, max_port=8100):
    """
    Scans a range of ports to bind to, ensuring error-free launch even under concurrent runs.
    """
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except socket.error:
                continue
    return start_port


class LiveVisualizerHandler(SimpleHTTPRequestHandler):
    conversation_id = "interactive_user_session"
    db_path = "data/cgm_memory.db"
    
    def log_message(self, format, *args):
        # Silence HTTP log prints in terminal to maintain clean aesthetic
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == '/api/graph':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_data = get_live_graph_json(self.conversation_id, self.db_path)
            self.wfile.write(json_data.encode('utf-8'))
        elif self.path == '/api/turns':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_data = get_turns_with_embeddings(self.conversation_id, self.db_path)
            self.wfile.write(json_data.encode('utf-8'))
        elif self.path == '/api/embeddings':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_data = get_2d_embeddings(self.conversation_id, self.db_path)
            self.wfile.write(json_data.encode('utf-8'))
        elif self.path == '/api/system':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_data = get_system_status()
            self.wfile.write(json_data.encode('utf-8'))
        elif self.path == '/api/logs':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_data = get_pipeline_logs()
            self.wfile.write(json_data.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")


def start_live_visualizer_server(conversation_id: str, db_path: str = "data/cgm_memory.db", port: int = 8050) -> int:
    """
    Launches visualizer server in a background daemon thread and pops open a browser window.
    """
    class ConfiguredHandler(LiveVisualizerHandler):
        pass
    ConfiguredHandler.conversation_id = conversation_id
    ConfiguredHandler.db_path = db_path
    
    actual_port = find_available_port(port)
    server = HTTPServer(("127.0.0.1", actual_port), ConfiguredHandler)
    
    def serve():
        server.serve_forever()
        
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    
    print(f"\n[Live Visualizer] Server successfully active at: http://localhost:{actual_port}/")
    print(f"[Live Visualizer] Launching automated web browser window...")
    
    # Non-blocking web browser launch
    webbrowser.open(f"http://localhost:{actual_port}/")
    return actual_port

if __name__ == "__main__":
    generate_graph_visualization()

# generate_VCMG.py
# -------------------------------------------------------------
# Generate VCMG (Vulnerability Cognition Mapping Graph) with VWS metrics
# from DOT and normalized C code files.
# This script processes DOT-format program graphs, merges node features,
# computes vulnerability-aware scores (VWS, VWS_C), and outputs .pkl files
# suitable for downstream machine learning tasks.
# Corresponds to the VCMG construction step described in our paper.
# -------------------------------------------------------------

import os
import re
import pickle
import argparse
import networkx as nx
from collections import defaultdict
from parse_dot import parse_dot  # Make sure this module exists and works correctly

def compute_vws_features(code):
    fc = len(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(', code))  # function calls
    au = len(re.findall(r'\[[^\]]*\]', code))                    # array usage
    pu = len(re.findall(r'\*\s*[a-zA-Z_]', code))                # pointer usage
    ae = len(re.findall(r'[\+\-\*/%]', code))                    # arithmetic expressions
    return fc, au, pu, ae

def merge_nodes_and_edges(graph_data, c_code_lines):
    merged_nodes = {}
    merged_edges = []
    for node in graph_data['nodes']:
        nid = node['id']
        line_no = node['line_number']
        if line_no in merged_nodes:
            merged_nodes[line_no]['ids'].append(nid)
        else:
            merged_nodes[line_no] = {
                'ids': [nid],
                'line_number': line_no,
                'degree_centrality': None,
                'closeness_centrality': None,
                'katz_centrality': None,
                'code_content': None,
                'vws_score': 0.0,
                'vws_score_centry': 0.0
            }

    if c_code_lines is not None:
        for line_no, data in merged_nodes.items():
            if 0 < line_no <= len(c_code_lines):
                data['code_content'] = c_code_lines[line_no - 1]

    full_graph = nx.DiGraph()
    for g_type in ['DDG', 'CDG', 'CFG']:
        edges = graph_data[g_type]['edges']
        G = nx.DiGraph()
        for edge in edges:
            src_id, tgt_id = edge['source'], edge['target']
            src_line = tgt_line = None
            for line, data in merged_nodes.items():
                if src_id in data['ids']:
                    src_line = data['line_number']
                if tgt_id in data['ids']:
                    tgt_line = data['line_number']
            if src_line and tgt_line:
                merged_edges.append((g_type, src_line, tgt_line))
                G.add_edge(src_line, tgt_line)
                full_graph.add_edge(src_line, tgt_line)

        # Compute centrality metrics
        deg_central = nx.degree_centrality(G)
        clo_central = nx.closeness_centrality(G)
        katz_central = nx.katz_centrality(G, max_iter=1000)
        for line, data in merged_nodes.items():
            data['degree_centrality'] = deg_central.get(line)
            data['closeness_centrality'] = clo_central.get(line)
            data['katz_centrality'] = katz_central.get(line)

    # Global metrics
    betweenness = nx.betweenness_centrality(full_graph, normalized=True)
    clustering = nx.clustering(full_graph.to_undirected())

    # VWS parameters
    α, β, γ, δ, λ = 1.0, 1.0, 1.0, 1.0, 1.0
    ω1, ω2, ω3 = 0.4, 0.3, 0.3
    k = 10

    for line, data in merged_nodes.items():
        code = data['code_content'] or ""
        FC = len(re.findall(r'\b(malloc|free|memcpy|memset|strcpy|scanf|gets|system|execl|FUN\d*)\b', code))
        AU = len(re.findall(r'\[.*?\]', code))
        PU = len(re.findall(r'\*', code))
        AE = len(re.findall(r'[\+\-\*/%]', code))
        vws = α * FC + β * AU + γ * PU + δ * AE
        data['vws_score'] = vws

        deg = data.get('degree_centrality') or 0.0
        btw = betweenness.get(line, 0.0)
        clu = clustering.get(line, 0.0)
        C = ω1 * deg + ω2 * btw + ω3 * clu
        vws_c = vws + λ * C * k
        data['vws_score_centry'] = vws_c

    return merged_nodes, merged_edges

def main():
    parser = argparse.ArgumentParser(description="Generate VWS-enriched PKL files from DOT and C code.")
    parser.add_argument("-d", "--dot_dir", required=True, help="Directory containing .dot files")
    parser.add_argument("-c", "--c_dir", required=True, help="Directory containing .c code files")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save output .pkl files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dot_files = [f for f in os.listdir(args.dot_dir) if f.endswith('.dot')]

    for dot_file in dot_files:
        try:
            func_name = os.path.splitext(dot_file)[0]
            graph_data = {
                'nodes': [],
                'vulnerable_nodes': [],
                'vulnerable_line': None,
                'AST': {'edges': []},
                'CFG': {'edges': []},
                'DDG': {'edges': []},
                'CDG': {'edges': []}
            }

            dot_path = os.path.join(args.dot_dir, dot_file)
            with open(dot_path, 'r') as f:
                dot_data = f.read()
                parse_dot(dot_data, graph_data)

            c_path = os.path.join(args.c_dir, f"{func_name}.c")
            if os.path.exists(c_path):
                with open(c_path, 'r') as f:
                    c_lines = f.readlines()
            else:
                print(f"[Warning] C file not found: {c_path}")
                c_lines = None

            merged_nodes, merged_edges = merge_nodes_and_edges(graph_data, c_lines)

            output_data = {
                'function_name': func_name,
                'merged_nodes': list(merged_nodes.values()),
                'merged_edges': merged_edges,
                'code_lines': c_lines
            }

            output_file = os.path.join(args.output_dir, f"{func_name}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)
            print(f"[OK] Saved: {output_file}")

        except Exception as e:
            print(f"[Error] Failed on {dot_file}: {e}")
            continue

if __name__ == "__main__":
    main()

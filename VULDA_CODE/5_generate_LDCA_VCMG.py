# generate_LDCA_VCMG.py
# --------------------------------------------------------------------
# Module: LDCA - Local Dependency Context Aggregation
# This script implements the LDCA module described in the paper,
# performing path fusion (CFG & DDG) and Soft Attention using VWS_C.
# It processes merged graph-based code representations and
# outputs VCMG vectors for downstream tasks.
# --------------------------------------------------------------------

import os
import pickle
import sent2vec
import argparse

def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]

def no_smooth(code_list, merged_edges):
    return code_list, [[i + 1] for i in range(len(code_list))]

def CFG_path_fusion(code_list, merged_edges):
    print("Code list length:", len(code_list))
    extend_code_list = []
    extend_line_map = []
    cfg_paths = []

    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'CFG':
            cfg_paths.append((source_line, target_line))

    for (a, b) in cfg_paths:
        for (b_next, c) in cfg_paths:
            if b_next == b:
                merged_code = code_list[a - 1] + code_list[b - 1] + code_list[c - 1]
                extend_code_list.append(merged_code)
                extend_line_map.append([a, b, c])

    return extend_code_list, extend_line_map

def DDG_path_fusion(code_list, merged_edges):
    print("Code list length:", len(code_list))
    extend_code_list = []
    extend_line_map = []
    ddg_nodes = {}

    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'DDG' and source_line != 1 and target_line != 1:
            if target_line not in ddg_nodes:
                ddg_nodes[target_line] = []
            ddg_nodes[target_line].append(source_line)

    for target_line, source_lines in ddg_nodes.items():
        code_fragments = [code_list[i - 1] for i in source_lines]
        merged_code = ''.join(code_fragments) + code_list[target_line - 1]
        extend_code_list.append(merged_code)
        extend_line_map.append(source_lines + [target_line])

    print("=" * 50)
    print("DDG-fused code list:", extend_code_list)
    print("DDG-fused code list length:", len(extend_code_list))
    print("=" * 50)

    return extend_code_list, extend_line_map

def path_fusion(code_list, merged_edges):
    print("Code list length:", len(code_list))
    extend_code_list = []
    extend_line_map = []

    cfg_paths = []
    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'CFG':
            cfg_paths.append((source_line, target_line))

    for (a, b) in cfg_paths:
        for (b_next, c) in cfg_paths:
            if b_next == b:
                merged_code = code_list[a - 1] + code_list[b - 1] + code_list[c - 1]
                extend_code_list.append(merged_code)
                extend_line_map.append([a, b, c])

    ddg_nodes = {}
    for graph_type, source_line, target_line in merged_edges:
        if graph_type == 'DDG' and source_line != 1 and target_line != 1:
            if target_line not in ddg_nodes:
                ddg_nodes[target_line] = []
            ddg_nodes[target_line].append(source_line)

    for target_line, source_lines in ddg_nodes.items():
        code_fragments = [code_list[i - 1] for i in source_lines]
        merged_code = ''.join(code_fragments) + code_list[target_line - 1]
        extend_code_list.append(merged_code)
        extend_line_map.append(source_lines + [target_line])

    return extend_code_list, extend_line_map

def process_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as pkl_file:
        edge_and_code_data = pickle.load(pkl_file)

    merged_nodes = edge_and_code_data['merged_nodes']
    merged_edges = list(set(edge_and_code_data['merged_edges']))
    sorted_nodes = sorted(merged_nodes, key=lambda node: node['line_number'])

    code_list = []
    vws_score_map = {}
    vws_c_map = {}  # Store VWS_C values

    for node in sorted_nodes:
        line_number = node['line_number']
        while len(code_list) + 1 < line_number:
            code_list.append(None)
        code_list.append(node['code_content'])
        vws_score_map[line_number] = node.get('vws_score_centry', 1.0)

        # Compute VWS_C value
        lambda_value = 0.5  # Example λ, adjust as needed
        omega1, omega2, omega3 = 0.3, 0.3, 0.4  # Example weights ω1, ω2, ω3
        degree = node.get('degree', 1)
        betweenness = node.get('betweenness', 1)
        clustering = node.get('clustering', 1)
        k = 1  # Example k value

        # VWS Ablation (commented)
        # VWS_C = vws_score_map[line_number] + lambda_value * (omega1 * degree + omega2 * betweenness + omega3 * clustering) * k
        VWS_C = vws_score_map[line_number]
        vws_c_map[line_number] = VWS_C

    # Select path fusion method
    extend_code_list, extend_line_map = path_fusion(code_list, merged_edges)
    # extend_code_list, extend_line_map = CFG_path_fusion(code_list, merged_edges)
    # extend_code_list, extend_line_map = no_smooth(code_list, merged_edges)
    # extend_code_list, extend_line_map = DDG_path_fusion(code_list, merged_edges)

    channel = []
    for merged_code, involved_lines in zip(extend_code_list, extend_line_map):
        if merged_code is not None:
            # 1. Get embedding vector of merged code snippet
            emb = sentence_embedding(merged_code)

            # 2. Get VWS_C value for each involved line
            scores = [vws_c_map.get(i, 1.0) for i in involved_lines]

            # 3. Calculate average weight
            avg_score = sum(scores) / len(scores) if scores else 1.0

            # 4. Calculate Soft Attention weights (normalized VWS_C)
            attention_weights = [score / avg_score for score in scores]

            # 5. Weight embedding vector by Soft Attention weights
            weighted_vec = emb * sum(attention_weights)

            # 6. Append weighted vector to channel list
            channel.append(weighted_vec)

    return channel

def main():
    parser = argparse.ArgumentParser(description="Generate VCMG channel vectors with LDCA module")
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing raw .pkl files')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to save processed vectors')
    parser.add_argument('-m', '--model_path', required=True, help='Path to trained sent2vec model (.bin)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(args.model_path)

    pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pkl")]
    for pkl_file in pkl_files:
        pkl_file_path = os.path.join(args.input_dir, pkl_file)
        print("Processing:", pkl_file_path)
        try:
            channel = process_pkl_file(pkl_file_path)
            print("Channel length:", len(channel))
            print("Vector dimension:", len(channel[0]) if channel else "No data")
            if channel:
                out_pkl = os.path.join(args.output_dir, pkl_file)
                with open(out_pkl, 'wb') as f:
                    pickle.dump(channel, f)
                print("Saved to:", out_pkl)
        except Exception as e:
            print("Error processing:", pkl_file_path, "Error message:", str(e))
            pass

if __name__ == "__main__":
    main()

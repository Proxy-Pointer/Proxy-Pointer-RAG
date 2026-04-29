"""
Proxy-Pointer MultiModal: Skeleton Tree Builder

Builds the structural tree from markdown documents and extracts figure paths.
(Image captioning is now handled by the LLM Synthesizer during retrieval).
"""
import os
import re
import json
import logging

def _extract_nodes_from_markdown(markdown_content: str):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    img_pattern = r'!\[(.*?)\]\((.*?)\)'
    
    node_list = []
    lines = markdown_content.split('\n')
    in_code_block = False
    global_fig_counter = 1

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        if re.match(code_block_pattern, stripped):
            in_code_block = not in_code_block
            continue

        if not in_code_block:
            m = re.match(header_pattern, stripped)
            if m:
                node_list.append({
                    'title': m.group(2).strip(),
                    'line_num': line_num,
                    'level': len(m.group(1)),
                    'figures': []
                })
            elif node_list:
                for img_match in re.finditer(img_pattern, stripped):
                    alt_text = img_match.group(1)
                    src = img_match.group(2)
                    
                    node_list[-1]['figures'].append({
                        "fig_id": f"fig_{global_fig_counter}",
                        "filename": src
                    })
                    global_fig_counter += 1

    return node_list, lines

def _build_tree_from_nodes(node_list):
    if not node_list: return []
    stack = []
    root_nodes = []
    counter = 1
    for node in node_list:
        level = node['level']
        tree_node = {'title': node['title'], 'node_id': str(counter).zfill(4), 'line_num': node['line_num'], 'figures': node.get('figures', []), 'nodes': []}
        counter += 1
        while stack and stack[-1][1] >= level: stack.pop()
        if not stack: root_nodes.append(tree_node)
        else:
            if 'nodes' not in stack[-1][0]: stack[-1][0]['nodes'] = []
            stack[-1][0]['nodes'].append(tree_node)
        stack.append((tree_node, level))
    return root_nodes

def _write_node_ids(data, node_id=1):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        if 'nodes' in data: node_id = _write_node_ids(data['nodes'], node_id)
    elif isinstance(data, list):
        for item in data: node_id = _write_node_ids(item, node_id)
    return node_id

def _format_structure(structure, order=None):
    if not order: return structure
    if isinstance(structure, dict):
        if 'nodes' in structure: structure['nodes'] = _format_structure(structure['nodes'], order)
        if not structure.get('nodes'): structure.pop('nodes', None)
        if not structure.get('figures'): structure.pop('figures', None)
        structure = {k: structure[k] for k in order if k in structure}
    elif isinstance(structure, list): structure = [_format_structure(item, order) for item in structure]
    return structure

def md_to_skeleton_tree(md_path: str) -> dict:
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    node_list, lines = _extract_nodes_from_markdown(content)
    tree = _build_tree_from_nodes(node_list)
    _write_node_ids(tree)
    tree = _format_structure(tree, order=['title', 'node_id', 'line_num', 'figures', 'nodes'])
    return {'doc_name': os.path.splitext(os.path.basename(md_path))[0], 'structure': tree}

def build_skeleton_trees(dataset_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            md_files = [f for f in os.listdir(item_path) if f.endswith(".md")]
            if md_files:
                md_file = os.path.join(item_path, md_files[0])
                tree_data = md_to_skeleton_tree(md_file)
                tree_data['doc_folder'] = item_path.replace("\\", "/")
                out_path = os.path.join(out_dir, f"{item}_structure.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(tree_data, f, indent=2, ensure_ascii=False)

def get_md_path_for_doc(dataset_dir, doc_id):
    for root, dirs, files in os.walk(dataset_dir):
        if f"{doc_id}.md" in files: return os.path.join(root, f"{doc_id}.md").replace("\\", "/")
    return None

if __name__ == "__main__":
    from src.config import DATASET_DIR, TREES_DIR
    build_skeleton_trees(DATASET_DIR, TREES_DIR)

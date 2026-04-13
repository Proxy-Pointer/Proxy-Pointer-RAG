"""
Proxy-Pointer: Skeleton Tree Builder

Builds a structural tree (title + node_id + line_num) from a Markdown file
by parsing heading hierarchy. This is a self-contained reimplementation of
the core PageIndex md_to_tree logic, stripped to only what we need:

  - No LLM calls (no summaries, no descriptions)
  - No text content embedded in nodes
  - No thinning / token-counting
  - Zero external dependencies beyond the Python stdlib

Usage:
    from src.indexing.build_skeleton_trees import build_skeleton_trees
"""
import os
import re
import json
import logging


# ── Step 1: Extract flat node list from Markdown headings ───────────────
def _extract_nodes_from_markdown(markdown_content: str):
    """
    Scan markdown for ATX headings (# through ######), ignoring those
    inside fenced code blocks.  Returns (node_list, lines).

    Each node: {'title': str, 'line_num': int (1-indexed)}
    """
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []

    lines = markdown_content.split('\n')
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        if re.match(code_block_pattern, stripped):
            in_code_block = not in_code_block
            continue

        if not stripped:
            continue

        if not in_code_block:
            m = re.match(header_pattern, stripped)
            if m:
                node_list.append({
                    'title': m.group(2).strip(),
                    'line_num': line_num,
                })

    return node_list, lines


# ── Step 2: Attach heading level + raw text span to each node ───────────
def _extract_node_text_content(node_list, markdown_lines):
    """
    For each node, determine its heading level and the text between it
    and the next heading (needed internally for tree construction, but
    stripped from the final output for skeleton mode).
    """
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)

        if header_match is None:
            logging.warning(
                f"Line {node['line_num']} is not a valid header: "
                f"'{line_content}'"
            )
            continue

        all_nodes.append({
            'title': node['title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1)),
        })

    # Compute text spans (start → next heading or EOF)
    for i, node in enumerate(all_nodes):
        start = node['line_num'] - 1
        end = (all_nodes[i + 1]['line_num'] - 1
               if i + 1 < len(all_nodes) else len(markdown_lines))
        node['text'] = '\n'.join(markdown_lines[start:end]).strip()

    return all_nodes


# ── Step 3: Convert flat list → nested tree via stack ───────────────────
def _build_tree_from_nodes(node_list):
    """
    Walk the flat node list and nest children under their parents based
    on heading level.  Assigns zero-padded node_ids sequentially.
    """
    if not node_list:
        return []

    stack = []          # (tree_node, level)
    root_nodes = []
    counter = 1

    for node in node_list:
        level = node['level']
        tree_node = {
            'title': node['title'],
            'node_id': str(counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': [],
        }
        counter += 1

        # Pop nodes that are at the same or deeper level
        while stack and stack[-1][1] >= level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            stack[-1][0]['nodes'].append(tree_node)

        stack.append((tree_node, level))

    return root_nodes


# ── Step 4: Re-assign sequential node_ids (post-order safe) ────────────
def _write_node_ids(data, node_id=1):
    """Recursively re-number node_ids in pre-order traversal (1-indexed)."""
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = _write_node_ids(data[key], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = _write_node_ids(item, node_id)
    return node_id


# ── Step 5: Format / strip unwanted fields ──────────────────────────────
def _format_structure(structure, order=None):
    """Reorder keys and prune empty `nodes` lists."""
    if not order:
        return structure

    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = _format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = {k: structure[k] for k in order if k in structure}

    elif isinstance(structure, list):
        structure = [_format_structure(item, order) for item in structure]

    return structure


# ── Public API ──────────────────────────────────────────────────────────
def md_to_skeleton_tree(md_path: str) -> dict:
    """
    Parse a Markdown file and return a skeleton structure tree.

    Returns:
        {
          "doc_name": "...",
          "line_count": N,
          "structure": [ {title, node_id, line_num, nodes?}, ... ]
        }

    No LLM calls are made.  No text content is retained in the output.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    line_count = content.count('\n') + 1
    node_list, lines = _extract_nodes_from_markdown(content)
    nodes_with_content = _extract_node_text_content(node_list, lines)
    tree = _build_tree_from_nodes(nodes_with_content)

    _write_node_ids(tree)

    # Skeleton mode: keep title, node_id, line_num — strip text
    tree = _format_structure(
        tree,
        order=['title', 'node_id', 'line_num', 'nodes'],
    )

    return {
        'doc_name': os.path.splitext(os.path.basename(md_path))[0],
        'line_count': line_count,
        'structure': tree,
    }


def build_skeleton_trees(data_dir: str, trees_dir: str):
    """
    For each .md file in data_dir, build a skeleton structure tree
    if one doesn't already exist in trees_dir.
    """
    os.makedirs(trees_dir, exist_ok=True)
    md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]

    if not md_files:
        logging.warning(f"No Markdown files found in {data_dir}")
        return

    built = 0
    skipped = 0

    for file in md_files:
        base_name = file.replace('.md', '')
        target_json = os.path.join(trees_dir, f"{base_name}_structure.json")

        if os.path.exists(target_json):
            skipped += 1
            continue

        md_path = os.path.join(data_dir, file)
        logging.info(f"Building skeleton tree: {file}...")

        try:
            tree_data = md_to_skeleton_tree(md_path)

            with open(target_json, 'w', encoding='utf-8') as f:
                json.dump(tree_data, f, indent=2, ensure_ascii=False)

            logging.info(f"  -> Tree mapped to: {target_json}")
            built += 1

        except Exception as e:
            logging.error(f"  -> Failed building tree for {file}: {e}")

    logging.info(f"Skeleton trees: {built} built, {skipped} already existed.")


# ── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.config import DATA_DIR, TREES_DIR

    build_skeleton_trees(str(DATA_DIR), str(TREES_DIR))

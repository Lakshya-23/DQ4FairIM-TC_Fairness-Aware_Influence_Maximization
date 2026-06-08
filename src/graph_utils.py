"""
graph_utils.py
--------------
Graph construction, dataset loading, and community assignment utilities.

Supports:
  • Synthetic Homophily-BA networks   (DQ4FairIM baseline)
  • Stochastic Block Model            (FAIRTCIM synthetic experiments)
  • SNAP Facebook Ego-Networks        (DQ4FairIM baseline)
"""

import os
import random
import numpy as np
import networkx as nx
from collections import defaultdict


# ──────────────────────────────────────────────────────────
#  Synthetic: Stochastic Block Model  (FAIRTCIM §6.1)
# ──────────────────────────────────────────────────────────

def build_sbm_graph(
    n: int = 300,
    group_ratio: float = 0.7,    # fraction in majority group
    p_within: float = 0.05,      # edge prob within group
    p_across: float = 0.005,     # edge prob across groups
    weight_range: tuple = (0.05, 0.3),
    seed: int = 42,
) -> tuple[nx.Graph, dict]:
    """
    Stochastic Block Model with 2 communities.
    Returns (graph, communities) where communities = {node -> group_label}.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    sizes = [int(n * group_ratio), n - int(n * group_ratio)]
    probs = [[p_within, p_across], [p_across, p_within]]

    G_sbm = nx.stochastic_block_model(sizes, probs, seed=seed)
    G     = nx.Graph()
    G.add_nodes_from(G_sbm.nodes())

    for u, v in G_sbm.edges():
        w = round(rng.uniform(*weight_range), 3)
        G.add_edge(u, v, weight=w)

    # community label: 0 = majority, 1 = minority
    communities = {}
    for node in G.nodes():
        communities[node] = 0 if node < sizes[0] else 1

    return G, communities


# ──────────────────────────────────────────────────────────
#  Synthetic: Homophily-BA  (DQ4FairIM §VI-A)
# ──────────────────────────────────────────────────────────

def build_hba_graph(
    n: int = 500,
    minority_ratio: float = 0.2,
    homophily: float = 0.8,
    m: int = 4,                  # edges per new node
    weight_range: tuple = (0.05, 0.3),
    seed: int = 42,
) -> tuple[nx.Graph, dict]:
    """
    Homophily-Barabási-Albert network.
    Minority nodes: label 1 (20%),  Majority: label 0 (80%).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    G = nx.Graph()
    communities = {}

    def assign_group(node_id):
        return 1 if rng.random() < minority_ratio else 0

    # seed graph
    for i in range(m + 1):
        G.add_node(i)
        communities[i] = assign_group(i)

    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=round(rng.uniform(*weight_range), 3))

    # grow
    for new_node in range(m + 1, n):
        new_group = assign_group(new_node)
        G.add_node(new_node)
        communities[new_node] = new_group

        existing = list(G.nodes())[:-1]
        degrees  = dict(G.degree())
        targets  = set()

        attempts = 0
        while len(targets) < m and attempts < m * 20:
            attempts += 1
            # preferential attachment weighted by homophily
            weights = []
            for nd in existing:
                if nd in targets:
                    weights.append(0)
                    continue
                h = homophily if communities[nd] == new_group else (1 - homophily)
                weights.append(h * (degrees.get(nd, 1)))

            total = sum(weights)
            if total == 0:
                break
            probs  = [w / total for w in weights]
            chosen = rng.choices(existing, weights=probs)[0]
            targets.add(chosen)

        for t in targets:
            G.add_edge(new_node, t,
                       weight=round(rng.uniform(*weight_range), 3))

    return G, communities



# ──────────────────────────────────────────────────────────
#  Real Dataset: SNAP Facebook Ego-Networks
#  Used in DQ4FairIM Table II as "Facebook [21]"
#  Source: snap.stanford.edu/data/ego-facebook.html
# ──────────────────────────────────────────────────────────

def load_facebook_snap(
    sample_size: int = 1000,
    n_samples: int = 1,
    seed: int = 42,
    weight_range: tuple = (0.01, 0.05),
) -> list:
    """
    Download and load the SNAP Facebook ego-network dataset.
    Extracts gender from anonymized feature vectors as the sensitive attribute,
    matching the exact procedure in DQ4FairIM §VI-A.

    Returns a list of (graph, communities) tuples — one per sampled subgraph.
    Each subgraph has `sample_size` nodes sampled via BFS from a random seed node.

    Falls back to SBM approximation if download fails.
    """
    import urllib.request, gzip, io, collections, tarfile

    rng = random.Random(seed)

    # ── Try downloading from SNAP ────────────────────────────────────────
    edge_url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    tar_url = "https://snap.stanford.edu/data/facebook.tar.gz"

    try:
        print("[Facebook] Downloading edge list from SNAP...")
        req = urllib.request.urlopen(edge_url, timeout=30)
        raw = gzip.decompress(req.read()).decode()
        edges_raw = [tuple(map(int, line.split()))
                     for line in raw.strip().split("\n") if line]

        G_full = nx.Graph()
        G_full.add_edges_from(edges_raw)
        print(f"[Facebook] Full graph: {G_full.number_of_nodes()} nodes, "
              f"{G_full.number_of_edges()} edges")

        # ── Extract gender from feature files via tar.gz ─────────────────────────
        node_gender = {}  # node_id -> 0 (male) or 1 (female)

        print("[Facebook] Downloading feature archive from SNAP...")
        req_tar = urllib.request.urlopen(tar_url, timeout=30)
        tar_bytes = io.BytesIO(req_tar.read())
        
        with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
            # Find all the base ego names (e.g., "facebook/0", "facebook/107")
            egos = set()
            for m in tar.getmembers():
                if m.name.endswith('.featnames'):
                    egos.add(m.name.replace('.featnames', ''))
            
            for ego_prefix in egos:
                try:
                    # 1. Read featnames to find the gender column index
                    fn_file = tar.extractfile(f"{ego_prefix}.featnames")
                    if not fn_file: continue
                    
                    fn_lines = fn_file.read().decode().strip().split("\n")
                    gender_idx = None
                    for i, line in enumerate(fn_lines):
                        if "gender" in line.lower():
                            gender_idx = i
                            break
                    
                    if gender_idx is None:
                        continue

                    # 2. Read the corresponding .feat file to extract the 1s and 0s
                    f_file = tar.extractfile(f"{ego_prefix}.feat")
                    if not f_file: continue
                    
                    f_lines = f_file.read().decode().strip().split("\n")
                    for line in f_lines:
                        parts = line.split()
                        if len(parts) > gender_idx + 1:
                            node_id = int(parts[0])
                            gender  = int(parts[gender_idx + 1])
                            node_gender[node_id] = gender
                except Exception:
                    continue

        print(f"[Facebook] Gender labels recovered: {len(node_gender)} nodes")

        # Nodes without gender label → assign by majority rule of neighbours
        for node in G_full.nodes():
            if node not in node_gender:
                nbr_genders = [node_gender[n] for n in G_full.neighbors(node)
                               if n in node_gender]
                node_gender[node] = (1 if sum(nbr_genders) > len(nbr_genders)/2
                                     else 0) if nbr_genders else 0

        # ── Sample subgraphs via BFS ─────────────────────────────────────
        all_nodes = list(G_full.nodes())
        result = []

        for i in range(n_samples):
            # BFS from a random starting node
            start = rng.choice(all_nodes)
            visited = []
            queue  = collections.deque([start])
            seen   = {start}
            while queue and len(visited) < sample_size:
                node = queue.popleft()
                visited.append(node)
                for nbr in G_full.neighbors(node):
                    if nbr not in seen:
                        seen.add(nbr); queue.append(nbr)

            if len(visited) < sample_size // 2:
                # BFS got stuck — fall back to random sampling
                visited = rng.sample(all_nodes, min(sample_size, len(all_nodes)))

            subgraph = G_full.subgraph(visited).copy()

            # Remap node IDs to 0..N-1
            mapping  = {n: i for i, n in enumerate(subgraph.nodes())}
            subgraph = nx.relabel_nodes(subgraph, mapping)

            communities = {mapping[n]: node_gender.get(n, 0) for n in visited}

            # Add edge weights
            for u, v in subgraph.edges():
                subgraph[u][v]["weight"] = round(rng.uniform(*weight_range), 4)

            result.append((subgraph, communities))

        minority_frac = sum(1 for v in result[0][1].values() if v==1) / len(result[0][1])
        print(f"[Facebook] Sampled {n_samples} subgraph(s) of {sample_size} nodes")
        print(f"[Facebook] Minority (female) fraction: {minority_frac:.1%}")
        return result

    except Exception as e:
        print(f"[Facebook] Download or parsing failed ({e}). Using SBM fallback.")
        result = []
        for i in range(n_samples):
            G, c = build_sbm_graph(
                n=sample_size, group_ratio=0.73,
                p_within=0.06, p_across=0.003, seed=seed+i
            )
            result.append((G, c))
        return result


def load_facebook_single(sample_size: int = 1000, seed: int = 42):
    """Convenience wrapper — returns (graph, communities) for a single sample."""
    results = load_facebook_snap(sample_size=sample_size, n_samples=1, seed=seed)
    return results[0]


# ──────────────────────────────────────────────────────────
#  Adjacency Matrix Helper
# ──────────────────────────────────────────────────────────

def get_adjacency_matrix(graph: nx.Graph,
                         weighted: bool = True) -> np.ndarray:
    """Return the adjacency matrix as a numpy array."""
    nodes = sorted(graph.nodes())
    n     = len(nodes)
    idx   = {nd: i for i, nd in enumerate(nodes)}
    adj   = np.zeros((n, n), dtype=np.float32)

    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0) if weighted else 1.0
        adj[idx[u], idx[v]] = w
        adj[idx[v], idx[u]] = w

    return adj


# ──────────────────────────────────────────────────────────
#  Graph Pool (for generalisation experiments, DQ4FairIM §VII-A)
# ──────────────────────────────────────────────────────────

def build_graph_pool(n_graphs: int = 10,
                     graph_type: str = "hba",
                     n_nodes: int = 300,
                     seed_start: int = 0,
                     **kwargs) -> list[tuple]:
    """
    Build a pool of graphs for training the RL agent.
    Each element is (graph, communities).
    """
    pool = []
    for i in range(n_graphs):
        s = seed_start + i
        if graph_type == "hba":
            g, c = build_hba_graph(n=n_nodes, seed=s, **kwargs)
        elif graph_type == "sbm":
            g, c = build_sbm_graph(n=n_nodes, seed=s, **kwargs)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        pool.append((g, c))
    return pool


# ──────────────────────────────────────────────────────────
#  Community Statistics
# ──────────────────────────────────────────────────────────

def community_stats(graph: nx.Graph, communities: dict) -> dict:
    """Print and return basic community statistics."""
    comm_counts = defaultdict(int)
    for node in graph.nodes():
        comm_counts[communities.get(node, "unknown")] += 1

    degrees = dict(graph.degree())
    comm_degree = defaultdict(list)
    for node, deg in degrees.items():
        comm_degree[communities.get(node, "unknown")].append(deg)

    stats = {}
    for comm, count in comm_counts.items():
        avg_deg = np.mean(comm_degree[comm]) if comm_degree[comm] else 0
        stats[comm] = {
            "size": count,
            "fraction": count / graph.number_of_nodes(),
            "avg_degree": round(avg_deg, 2),
        }

    print("\n── Community Statistics ──────────────")
    for comm, s in stats.items():
        print(f"  Group {comm}: size={s['size']} "
              f"({s['fraction']:.1%}), avg_degree={s['avg_degree']}")
    # set community as node attribute for assortativity calculation
    nx.set_node_attributes(graph, communities, name="community")
    try:
        h_idx = nx.attribute_assortativity_coefficient(graph, "community")
        print(f"  Homophily index: {h_idx:.3f}")
    except Exception:
        print("  Homophily index: N/A")
    print("──────────────────────────────────────\n")
    return stats
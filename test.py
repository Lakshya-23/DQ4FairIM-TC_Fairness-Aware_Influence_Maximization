import torch, networkx as nx, numpy as np
print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.is_available()}')
print(f'NetworkX {nx.__version__} | NumPy: {np.__version__}')

import sys
sys.path.insert(0,'fairim')
from src import (
    build_hba_graph, build_sbm_graph, build_graph_pool,
    load_facebook_snap, load_facebook_single,
    simulate_ic_communities, simulate_tc_ic,
    FairIMEnvironment, DQNAgent,
    get_adjacency_matrix, community_stats,
    evaluate_all_baselines,
)
from src.diffusion import gini_coefficient
print('All imports OK.')


# ─── Dataset settings ───────────────────────────────────────
DATASET          = 'facebook'  # 'facebook' (real) or 'hba' (synthetic)
SAMPLE_SIZE      = 1000        # nodes per subgraph (DQ4FairIM uses 1000)
N_TRAIN_GRAPHS   = 6           # training subgraphs  (paper uses 6 train, 4 test)

# ─── IM problem ─────────────────────────────────────────────
BUDGET           = 10
DEADLINE         = None
IC_PROB          = 0.1
NUM_SIM_TRAIN    = 30
NUM_SIM_EVAL     = 300

# ─── Reward weights ─────────────────────────────────────────
PHI              = 1.0    # maximin fairness weight
LAMBDA_DISPARITY = 0.5    # disparity penalty (new in v3)
GAMMA_LATENCY    = 0.5

# ─── Dynamic graph settings ─────────────────────────────────
TEMPORAL_CHANGE  = 0.02
STEP_CHANGE      = 0.005

# ─── Agent ──────────────────────────────────────────────────
EMBED_DIM        = 64
S2V_ITERS        = 4
LR               = 1e-3
N_EPISODES       = 400
LOG_EVERY        = 50

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device  : {DEVICE}')
print(f'Dataset : {DATASET}')
print(f'Reward  : reach + {PHI}*maximin − {LAMBDA_DISPARITY}*disparity')


if DATASET == 'facebook':
    # Download SNAP Facebook ego-network (same dataset as DQ4FairIM Table II)
    # Downloads edge list + gender feature files from snap.stanford.edu
    print('Loading Facebook ego-network from SNAP...')
    print('  (requires internet — falls back to SBM if unavailable)')
    all_samples = load_facebook_snap(
        sample_size  = SAMPLE_SIZE,
        n_samples    = N_TRAIN_GRAPHS + 4,  # 6 train + 4 test
        seed         = 42,
    )
    pool          = all_samples[:N_TRAIN_GRAPHS]   # training graphs
    test_samples  = all_samples[N_TRAIN_GRAPHS:]   # held-out test graphs
else:
    # Synthetic Homophily-BA (original v3 default)
    pool = build_graph_pool(n_graphs=N_TRAIN_GRAPHS, graph_type='hba',
                           n_nodes=SAMPLE_SIZE, seed_start=0,
                           homophily=0.75, minority_ratio=0.15)
    test_samples = [build_hba_graph(n=SAMPLE_SIZE, seed=999,
                                    homophily=0.75, minority_ratio=0.15)]

g0, c0 = pool[0]
print(f'\nTraining pool: {len(pool)} graphs')
community_stats(g0, c0)


feature_dim = 3 + len(set(c0.values()))
print(f'Feature dim: {feature_dim}')
agent = DQNAgent(
    feature_dim=feature_dim, embed_dim=EMBED_DIM,
    n_s2v_iters=S2V_ITERS, lr=LR,
    gamma=1.0, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.992,
    memory_size=2000, batch_size=32,
    update_every=1, target_update_freq=50, device=DEVICE,
)
print(f'Agent ready (all 5 weight matrices trainable)')

# # Training Section - Uncomment when you need to train the agent (takes ~20-30 minutes on GPU)

# import time
# from collections import defaultdict

# history = defaultdict(list)
# t0 = time.time()
# print(f'Training {N_EPISODES} episodes | k={BUDGET} | phi={PHI}')
# print(f'Dynamics: temporal={TEMPORAL_CHANGE}/ep, step={STEP_CHANGE}/step')

# sim_fn = simulate_tc_ic if DEADLINE else simulate_ic_communities
# tc_kw  = {'deadline': DEADLINE} if DEADLINE else {}

# for ep in range(1, N_EPISODES+1):
#     g,c = pool[(ep-1) % len(pool)]
#     env = FairIMEnvironment(
#         base_graph=g, communities=c, budget=BUDGET,
#         deadline=DEADLINE,
#         temporal_change_rate=TEMPORAL_CHANGE,
#         step_change_rate=STEP_CHANGE,
#         ic_prob=IC_PROB, num_sim=NUM_SIM_TRAIN,
#         phi=PHI, lambda_disparity=LAMBDA_DISPARITY,
#         gamma_latency=GAMMA_LATENCY,
#     )
#     state=env.reset(); adj=get_adjacency_matrix(env.graph)
#     ep_r=0.0; done=False
#     while not done:
#         avail=env.available_actions()
#         action=agent.select_action(state,adj,avail)
#         nstate,reward,done,_=env.step(action)
#         nadj=get_adjacency_matrix(env.graph)  # updated after step perturbs graph
#         agent.store(state,adj,action,float(reward),nstate,nadj,done)
#         agent.update()
#         state=nstate; adj=nadj; ep_r+=float(reward)
#     final=sim_fn(env.graph,env.seed_set,c,prob=IC_PROB,num_simulations=40,**tc_kw)
#     history['ep'].append(ep)
#     history['reward'].append(ep_r)
#     history['outreach'].append(final['outreach'])
#     history['fairness'].append(final['fairness'])
#     history['disparity'].append(final['disparity'])
#     history['epsilon'].append(agent.epsilon)
#     if ep % LOG_EVERY == 0:
#         import numpy as _np
#         w=min(LOG_EVERY,ep)
#         print(f'  Ep {ep:4d}/{N_EPISODES} | eps={agent.epsilon:.3f} | '
#               f'Out={_np.mean(history["outreach"][-w:]):.4f} | '
#               f'Fair={_np.mean(history["fairness"][-w:]):.4f} | '
#               f'Disp={_np.mean(history["disparity"][-w:]):.4f} | '
#               f't={time.time()-t0:.0f}s')
#     # Decay epsilon once per episode (not per step)
#     # Rate 0.9851 → reaches eps_min=0.05 at episode ~200
#     agent.epsilon = max(0.05, agent.epsilon * 0.9851)

# print(f'\nDone in {time.time()-t0:.1f}s')
# agent.save('fairim/checkpoints/agent_v3.pt')
# print('Checkpoint saved.')

# #Training curve visualization (uncomment if you want to see the training curves)

# import matplotlib.pyplot as plt, numpy as np

# def smooth(lst,w=15): return [np.mean(lst[max(0,i-w):i+1]) for i in range(len(lst))]

# fig,axes=plt.subplots(1,3,figsize=(14,4))
# fig.suptitle(f'Training curves v3 (phi={PHI}, step_change={STEP_CHANGE}, k={BUDGET})',fontsize=12)
# for ax,(key,label,col) in zip(axes,[
#     ('outreach','Outreach','#378add'),
#     ('fairness','Fairness','#1d9e75'),
#     ('disparity','Disparity','#d85a30')]):
#     raw=history[key]; smo=smooth(raw); eps=history['ep']
#     ax.fill_between(eps,np.array(smo)-np.std(raw)*.2,
#                     np.array(smo)+np.std(raw)*.2,alpha=.1,color=col)
#     ax.plot(eps,raw,color=col,alpha=.2,lw=.8)
#     ax.plot(eps,smo,color=col,lw=2.0)
#     ax.set_title(label,fontsize=11); ax.set_xlabel('Episode'); ax.grid(alpha=.2)
#     ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig('fairim/results/training_v3.png',dpi=150,bbox_inches='tight')
# plt.show()



# Use held-out test graph (real Facebook subgraph or synthetic)
test_graph, test_comm = test_samples[1]
print(f'Test graph: {test_graph.number_of_nodes()} nodes, '
      f'{test_graph.number_of_edges()} edges')
community_stats(test_graph, test_comm)

# ── Helper: run any seed-selection method on a continuously evolving graph ──
def eval_on_dynamic(seed_fn, graph, comm, k, step_change, ic_prob, num_sim):
    """
    seed_fn: callable(graph, k, communities) -> set of seed nodes
    Evaluates after re-perturbing the graph once per seed selection,
    matching the exact dynamic conditions the agent is trained on.
    """
    env_tmp = FairIMEnvironment(graph, comm, budget=k,
                                temporal_change_rate=0.0,
                                step_change_rate=step_change,
                                ic_prob=ic_prob, num_sim=num_sim)
    env_tmp.reset()
    # Baselines choose all seeds at once on initial graph snapshot
    seeds = seed_fn(env_tmp.graph, k, comm)
    # Simulate diffusion on the evolved graph (after k perturbations)
    evolved = env_tmp.graph.copy()
    for _ in range(k):
        evolved = env_tmp._apply_temporal_change(evolved, rate=step_change)
    return simulate_ic_communities(evolved, seeds, comm,
                                   prob=ic_prob, num_simulations=num_sim)

print('Evaluating on continuous dynamic graph (step_change={})...'.format(STEP_CHANGE))
from src.baselines import degree_seeding, pagerank_seeding, parity_seeding, fair_pagerank_seeding

results = {}
for name, fn in [
    ('Degree',       lambda g,k,c: degree_seeding(g,k)),
    ('PageRank',     lambda g,k,c: pagerank_seeding(g,k)),
    ('Parity',       parity_seeding),
    ('Fair-PageRank',fair_pagerank_seeding),
]:
    print(f'  {name}...')
    results[name] = eval_on_dynamic(fn, test_graph, test_comm,
                                    BUDGET, STEP_CHANGE, IC_PROB, NUM_SIM_EVAL)

# Agent: trained for continuous dynamics, evaluated the same way
print('  DQ4FairIM-TC...')
agent.epsilon = 0.0
env_eval = FairIMEnvironment(test_graph, test_comm, budget=BUDGET,
                             temporal_change_rate=0.0,
                             step_change_rate=STEP_CHANGE,
                             ic_prob=IC_PROB, num_sim=50,
                             phi=PHI, lambda_disparity=LAMBDA_DISPARITY,
                             gamma_latency=GAMMA_LATENCY)
state=env_eval.reset(); adj=get_adjacency_matrix(env_eval.graph); done=False
while not done:
    avail=env_eval.available_actions(); a=agent.select_action(state,adj,avail)
    state,_,done,_=env_eval.step(a); adj=get_adjacency_matrix(env_eval.graph)
results['DQ4FairIM-TC']=simulate_ic_communities(
    env_eval.graph, env_eval.seed_set, test_comm,
    prob=IC_PROB, num_simulations=NUM_SIM_EVAL)

print()
hdr=f'{"Method":<22}{"Outreach":>10}{"Fairness":>10}{"Disparity":>10}{"Gini":>8}'
print('─'*len(hdr)); print(hdr); print('─'*len(hdr))
for m,r in results.items():
    g_val=gini_coefficient(r.get('per_comm',{}))
    tag=' ◄' if m=='DQ4FairIM-TC' else ''
    print(f'  {m:<20}{r["outreach"]:>10.4f}{r["fairness"]:>10.4f}'
          f'{r["disparity"]:>10.4f}{g_val:>8.4f}{tag}')
print('─'*len(hdr))


import matplotlib.pyplot as plt, numpy as np

methods=list(results.keys()); AGENT='DQ4FairIM-TC'
fig,axes=plt.subplots(1,4,figsize=(16,4.5))
fig.suptitle(f'Method comparison — continuous dynamic graph (step_change={STEP_CHANGE})',fontsize=12)
for ax,(key,title,base_col) in zip(axes,[
    ('outreach','Outreach','#b5d4f4'),
    ('fairness','Maximin fairness','#9fe1cb'),
    ('disparity','Disparity','#f5c4b3'),
    (None,'Gini coefficient','#fac775')]):
    vals=[gini_coefficient(results[m].get('per_comm',{})) if key is None
          else results[m][key] for m in methods]
    bar_colors=['#1d9e75' if m==AGENT else base_col for m in methods]
    bars=ax.bar(methods,vals,color=bar_colors,width=0.6,edgecolor='none')
    ax.set_title(title,fontsize=11)
    ax.set_xticklabels([m.replace('-TC','\n-TC').replace('Fair-','Fair\n') for m in methods],fontsize=8)
    ax.grid(axis='y',alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,
                f'{val:.4f}',ha='center',va='bottom',fontsize=7)
plt.tight_layout()
plt.savefig('fairim/results/comparison_v3.png',dpi=150,bbox_inches='tight')
plt.show()


step_rates=[0.0,0.002,0.005,0.01,0.02,0.05]
temp_results=[]; baseline_seed=None
print(f'{"Step Rate":>10}{"Outreach":>10}{"Fairness":>10}{"Disparity":>10}{"DSeed%":>8}')
print('─'*52)
for rate in step_rates:
    env_t=FairIMEnvironment(test_graph,test_comm,budget=BUDGET,
                            temporal_change_rate=0.0,step_change_rate=rate,
                            ic_prob=IC_PROB,num_sim=50,phi=PHI,
                            lambda_disparity=LAMBDA_DISPARITY,gamma_latency=GAMMA_LATENCY)
    st=env_t.reset(); adj_t=get_adjacency_matrix(env_t.graph); d=False
    while not d:
        av=env_t.available_actions(); a=agent.select_action(st,adj_t,av)
        st,_,d,_=env_t.step(a); adj_t=get_adjacency_matrix(env_t.graph)
    res_t=simulate_ic_communities(env_t.graph,env_t.seed_set,test_comm,
                                  prob=IC_PROB,num_simulations=200)
    delta=0.0 if baseline_seed is None else \
          len(baseline_seed-env_t.seed_set)/max(len(baseline_seed),1)*100
    if baseline_seed is None: baseline_seed=set(env_t.seed_set)
    temp_results.append({'rate':rate,'outreach':res_t['outreach'],
                         'fairness':res_t['fairness'],'disparity':res_t['disparity'],'ds':delta})
    print(f'{rate:>10.3f}{res_t["outreach"]:>10.4f}{res_t["fairness"]:>10.4f}'
          f'{res_t["disparity"]:>10.4f}{delta:>8.1f}%')

import matplotlib.pyplot as plt
rates=[r['rate'] for r in temp_results]
fig,ax=plt.subplots(figsize=(8,4))
for key,col,lbl in [('outreach','#378add','Outreach'),
                     ('fairness','#1d9e75','Fairness'),
                     ('disparity','#d85a30','Disparity')]:
    ax.plot(rates,[r[key] for r in temp_results],'o-',color=col,lw=2,label=lbl)
ax.set_xlabel('Step change rate (edges perturbed per seed selection)')
ax.set_ylabel('Score'); ax.set_title('Resilience to continuous graph dynamics',fontsize=11)
ax.legend(); ax.grid(alpha=.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fairim/results/temporal_v3.png',dpi=150,bbox_inches='tight')
plt.show()


# Build the complete visualization cell
# Architecture:
# 1. Record a full episode step-by-step (capture graph state, seed set, IC spread at each step)
# 2. Layout the graph with community-aware positioning (spring layout, communities clustered)
# 3. Animate: edges added/removed highlighted, seeds pulsing, IC spread propagating, outliers marked


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE GRAPH VISUALIZATION
#  Shows: edge dynamics, seed selection, IC spread propagation, outliers
#  Uses matplotlib animation — runs inline in Colab
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML
import random as _random

# ── Step 1: Record one full episode ─────────────────────────────────────────
print("Recording episode for visualization...")

VIZ_GRAPH_SIZE = min(80, test_graph.number_of_nodes())  # keep it readable

# Sample a small connected subgraph for clear visualization
def sample_connected_subgraph(G, comm, target=VIZ_GRAPH_SIZE, seed=42):
    rng = _random.Random(seed)
    # Start from a node with both community types nearby
    bridge_nodes = [n for n in G.nodes()
                    if any(comm.get(nb, 0) != comm.get(n, 0) for nb in G.neighbors(n))]
    start = rng.choice(bridge_nodes) if bridge_nodes else rng.choice(list(G.nodes()))
    visited = []; queue = [start]; seen = {start}
    while queue and len(visited) < target:
        node = queue.pop(0); visited.append(node)
        for nb in sorted(G.neighbors(node)):
            if nb not in seen: seen.add(nb); queue.append(nb)
    sub = G.subgraph(visited).copy()
    mapping = {n: i for i, n in enumerate(sorted(sub.nodes()))}
    sub = nx.relabel_nodes(sub, mapping)
    comm_sub = {mapping[n]: comm.get(n, 0) for n in sorted(G.subgraph(visited).nodes())}
    return sub, comm_sub

viz_graph_base, viz_comm = sample_connected_subgraph(test_graph, test_comm)
N_VIZ = viz_graph_base.number_of_nodes()
print(f"Visualization subgraph: {N_VIZ} nodes, {viz_graph_base.number_of_edges()} edges")

# ── Fixed layout: community-clustered positions ──────────────────────────────
def community_layout(G, comm, seed=42):
    """Spring layout with community attraction — keeps groups clustered."""
    pos = nx.spring_layout(G, seed=seed, k=2.5/np.sqrt(len(G)), iterations=80)
    # Nudge nodes toward community centroid
    for group in [0, 1]:
        nodes = [n for n, c in comm.items() if c == group]
        if not nodes: continue
        cx = np.mean([pos[n][0] for n in nodes])
        cy = np.mean([pos[n][1] for n in nodes])
        offset = 0.6 if group == 0 else -0.6
        for n in nodes:
            pos[n] = np.array([pos[n][0]*0.6 + (cx+offset)*0.4,
                                pos[n][1]*0.6 + cy*0.4])
    return pos

pos = community_layout(viz_graph_base, viz_comm)

# ── Run a real agent episode on the viz graph ────────────────────────────────
viz_env = FairIMEnvironment(
    viz_graph_base, viz_comm, budget=BUDGET,
    temporal_change_rate=TEMPORAL_CHANGE,
    step_change_rate=STEP_CHANGE,
    ic_prob=IC_PROB, num_sim=20,
    phi=PHI, lambda_disparity=LAMBDA_DISPARITY, gamma_latency=GAMMA_LATENCY,
)

agent.epsilon = 0.0  # greedy

# Capture frames: each frame = state after one seed selection
frames = []
state = viz_env.reset()
adj   = get_adjacency_matrix(viz_env.graph)

# Frame 0: initial graph, no seeds
frames.append({
    'graph':       viz_env.graph.copy(),
    'seeds':       set(),
    'new_seed':    None,
    'added_edges': set(),
    'removed_edges': set(),
    'ic_reached':  set(),
    'step':        0,
    'fairness':    0.0,
    'disparity':   0.0,
    'outreach':    0.0,
})

prev_edges = set(viz_env.graph.edges())
done = False
step = 0

while not done:
    avail  = viz_env.available_actions()
    action = agent.select_action(state, adj, avail)
    state, reward, done, info = viz_env.step(action)
    adj_n  = get_adjacency_matrix(viz_env.graph)
    step  += 1

    cur_edges  = set(viz_env.graph.edges())
    added      = cur_edges - prev_edges
    removed    = prev_edges - cur_edges

    # Simulate IC spread from current seed set
    ic_res   = simulate_ic_communities(
        viz_env.graph, viz_env.seed_set, viz_comm,
        prob=IC_PROB, num_simulations=30)

    # Approximate which nodes are reached (threshold on per-node marginal)
    # Use multiple MC runs to estimate per-node activation probability
    reached = set()
    for _ in range(20):
        activated = set(viz_env.seed_set)
        frontier  = set(viz_env.seed_set)
        while frontier:
            nf = set()
            for u in frontier:
                for v in viz_env.graph.neighbors(u):
                    if v not in activated:
                        if _random.random() < IC_PROB:
                            activated.add(v); nf.add(v)
            frontier = nf
        reached |= activated  # union across runs = "ever reached"

    frames.append({
        'graph':         viz_env.graph.copy(),
        'seeds':         set(viz_env.seed_set),
        'new_seed':      action,
        'added_edges':   added,
        'removed_edges': removed,
        'ic_reached':    reached,
        'step':          step,
        'fairness':      ic_res['fairness'],
        'disparity':     ic_res['disparity'],
        'outreach':      ic_res['outreach'],
        'reward':        float(reward),
    })
    prev_edges = cur_edges
    adj = adj_n

print(f"Recorded {len(frames)} frames  ({len(frames)-1} seed selections + initial state)")

# ── Step 2: Identify outliers (isolated or low-degree nodes) ────────────────
degrees = dict(viz_graph_base.degree())
deg_vals = sorted(degrees.values())
deg_q25  = np.percentile(deg_vals, 15)
outlier_nodes = {n for n, d in degrees.items() if d <= deg_q25}
print(f"Outlier nodes (degree ≤ p15 = {deg_q25:.0f}): {len(outlier_nodes)}")

# ── Step 3: Animate ──────────────────────────────────────────────────────────
COLORS = {
    'majority':      '#4393c3',   # blue — majority community
    'minority':      '#d6604d',   # red  — minority community
    'seed':          '#f4a000',   # gold — seed node
    'new_seed':      '#ff4500',   # orange-red — just-selected seed
    'reached':       '#74c476',   # green — IC reached
    'unreached':     '#bbbbbb',   # gray  — not yet reached
    'outlier':       '#9e5b9e',   # purple — structural outlier
    'edge_normal':   '#cccccc',
    'edge_added':    '#2ca02c',   # green — new edge
    'edge_removed':  '#d62728',   # red — deleted edge
    'edge_seed':     '#f4a000',   # gold — edge from seed
}

fig = plt.figure(figsize=(16, 9), facecolor='#0f0f1a')
ax_main = fig.add_axes([0.0, 0.15, 0.65, 0.82], facecolor='#0f0f1a')
ax_fair  = fig.add_axes([0.67, 0.55, 0.30, 0.38], facecolor='#1a1a2e')
ax_disp  = fig.add_axes([0.67, 0.10, 0.30, 0.38], facecolor='#1a1a2e')
fig.patch.set_facecolor('#0f0f1a')

for ax in [ax_main, ax_fair, ax_disp]:
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

ax_fair.set_title('Fairness (maximin)', color='#74c476', fontsize=9, pad=4)
ax_disp.set_title('Disparity', color='#d6604d', fontsize=9, pad=4)
for ax in [ax_fair, ax_disp]:
    ax.set_xlim(0, len(frames))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2, color='#333344')
    ax.set_xlabel('Step', color='#888888', fontsize=8)

fair_line, = ax_fair.plot([], [], color='#74c476', lw=2)
disp_line, = ax_disp.plot([], [], color='#d6604d', lw=2)
fair_data, disp_data, step_data = [], [], []

# Pre-build node list in fixed order
nodes_sorted = sorted(viz_graph_base.nodes())

def get_node_color(n, frame):
    seeds    = frame['seeds']
    new_seed = frame.get('new_seed')
    reached  = frame.get('ic_reached', set())
    if n == new_seed:         return COLORS['new_seed']
    if n in seeds:            return COLORS['seed']
    if n in reached:          return COLORS['reached']
    if n in outlier_nodes:    return COLORS['outlier']
    return COLORS['majority'] if viz_comm.get(n, 0) == 0 else COLORS['minority']

def get_node_size(n, frame):
    if n == frame.get('new_seed'): return 280
    if n in frame['seeds']:        return 220
    if n in outlier_nodes:         return 60
    return 90

def get_node_alpha(n, frame):
    if n in frame['seeds'] or n == frame.get('new_seed'): return 1.0
    if n in frame.get('ic_reached', set()):               return 0.90
    return 0.55

def draw_frame(frame_idx):
    ax_main.cla()
    ax_main.set_facecolor('#0f0f1a')
    ax_main.axis('off')

    fr = frames[frame_idx]
    G  = fr['graph']

    # Draw edges by type
    all_edges   = list(G.edges())
    added       = fr.get('added_edges', set())
    removed_set = fr.get('removed_edges', set())  # shown as ghost
    seed_edges  = [(u,v) for u,v in all_edges if u in fr['seeds'] or v in fr['seeds']]
    normal_edges= [(u,v) for u,v in all_edges if (u,v) not in added and (u,v) not in seed_edges]
    add_edges   = [(u,v) for u,v in all_edges if (u,v) in added or (v,u) in added]

    def draw_edges(elist, color, alpha, width, style='solid'):
        coords = [(pos.get(u), pos.get(v)) for u,v in elist
                  if pos.get(u) is not None and pos.get(v) is not None]
        for (p1, p2) in coords:
            ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         color=color, alpha=alpha, lw=width,
                         linestyle=style, zorder=1)

    draw_edges(normal_edges, COLORS['edge_normal'], 0.25, 0.6)
    draw_edges(seed_edges,   COLORS['edge_seed'],   0.55, 1.2)
    draw_edges(add_edges,    COLORS['edge_added'],  0.85, 1.8)

    # Draw nodes
    valid_nodes = [n for n in nodes_sorted if n in G.nodes() and n in pos]
    xs    = [pos[n][0] for n in valid_nodes]
    ys    = [pos[n][1] for n in valid_nodes]
    cols  = [get_node_color(n, fr) for n in valid_nodes]
    sizes = [get_node_size(n, fr)  for n in valid_nodes]
    alphs = [get_node_alpha(n, fr) for n in valid_nodes]

    for x, y, c, s, a in zip(xs, ys, cols, sizes, alphs):
        ax_main.scatter(x, y, c=c, s=s, alpha=a, zorder=3, linewidths=0.5,
                        edgecolors='white' if a > 0.8 else 'none')

    # Highlight new seed with a ring
    if fr.get('new_seed') is not None and fr['new_seed'] in pos:
        px, py = pos[fr['new_seed']]
        circle = plt.Circle((px, py), 0.045, fill=False,
                             edgecolor='#ffffff', linewidth=2.5, zorder=5,
                             transform=ax_main.transData)
        ax_main.add_patch(circle)

    # Label seeds
    for n in fr['seeds']:
        if n in pos:
            ax_main.annotate(f'S{list(fr["seeds"]).index(n)+1}',
                              pos[n], fontsize=6.5, ha='center', va='center',
                              color='black', fontweight='bold', zorder=6)

    # Stats overlay
    n_reached  = len(fr.get('ic_reached', set()))
    n_seeds    = len(fr['seeds'])
    n_added    = len(fr.get('added_edges', set()))
    step_txt   = f"Step {fr['step']}/{BUDGET}"
    stats_txt  = (f"Seeds: {n_seeds}/{BUDGET}  |  "
                  f"Reached: {n_reached}/{N_VIZ} ({n_reached/max(N_VIZ,1):.0%})  |  "
                  f"Edges added: {n_added}  |  "
                  f"Fair: {fr['outreach']:.3f}  Disp: {fr['disparity']:.3f}")

    ax_main.text(0.01, 0.99, step_txt, transform=ax_main.transAxes,
                 fontsize=13, color='white', va='top', fontweight='bold')
    ax_main.text(0.01, 0.93, stats_txt, transform=ax_main.transAxes,
                 fontsize=8, color='#aaaaaa', va='top')

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS['majority'],  label='Majority (label 0)'),
        mpatches.Patch(color=COLORS['minority'],  label='Minority (label 1)'),
        mpatches.Patch(color=COLORS['seed'],      label='Seed node'),
        mpatches.Patch(color=COLORS['new_seed'],  label='New seed (this step)'),
        mpatches.Patch(color=COLORS['reached'],   label='IC reached'),
        mpatches.Patch(color=COLORS['outlier'],   label='Outlier (low degree)'),
        mpatches.Patch(color=COLORS['edge_added'],label='New edge (dynamic)'),
    ]
    ax_main.legend(handles=legend_items, loc='lower left',
                   fontsize=7, facecolor='#1a1a2e', edgecolor='#333344',
                   labelcolor='white', framealpha=0.9, ncol=2)

    # Metric charts
    if fr['step'] > 0:
        step_data.append(fr['step'])
        fair_data.append(fr['fairness'])
        disp_data.append(fr['disparity'])
        fair_line.set_data(step_data, fair_data)
        disp_line.set_data(step_data, disp_data)
        ax_fair.set_xlim(0, max(BUDGET+1, len(step_data)+1))
        ax_disp.set_xlim(0, max(BUDGET+1, len(step_data)+1))

def animate(i):
    draw_frame(i)
    return []

fig.suptitle('DQ4FairIM-TC — Live Graph Dynamics',
             color='white', fontsize=14, fontweight='bold', y=0.99)

ani = animation.FuncAnimation(
    fig, animate,
    frames=len(frames),
    interval=1400,     # ms between frames
    repeat=True,
    blit=False,
)


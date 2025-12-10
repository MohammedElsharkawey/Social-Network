"""
Assignment 2
Mohammed Yasser El.Sharkawey
2205149
"""

import os
import random
import gzip
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_graph(filepath="facebook_combined.txt"):
    print(f" Loading data from {filepath}...")
    if filepath.endswith(".gz"):
        with gzip.open(filepath, 'rt') as f:
            G = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=int)
    else:
        G = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
    
    print(f" Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def extract_features(G):
    """
    Removed 'Community' feature to make the model susceptible to attacks.
    Only using: Degree, Clustering, PageRank.
    """
    print("⏳ Extracting features (Degree, Clustering, PageRank)...")
    
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, alpha=0.85)
    
   
    
    nodes = list(G.nodes())
    features = []
    
    for n in nodes:
        features.append([
            degrees.get(n, 0),
            clustering.get(n, 0),
            pagerank.get(n, 0)
        ])
        
    return nodes, np.array(features)


def inject_bots(G, fraction=0.05):
    G_mixed = G.copy()
    num_existing = G.number_of_nodes()
    num_bots = int(num_existing * fraction)
    
    start_id = max(G.nodes()) + 1
    bot_nodes = [start_id + i for i in range(num_bots)]
    
    labels = {n: 0 for n in G.nodes()}
    
    existing_nodes = list(G.nodes())
    
    for bot in bot_nodes:
        G_mixed.add_node(bot)
        labels[bot] = 1 
        
        targets = random.sample(existing_nodes, 10) 
        for t in targets:
            G_mixed.add_edge(bot, t)
            
    for i in range(len(bot_nodes)):
        peer = bot_nodes[(i + 1) % len(bot_nodes)]
        G_mixed.add_edge(bot_nodes[i], peer)
        
    print(f"🤖 Injected {len(bot_nodes)} Bots (Baseline).")
    return G_mixed, bot_nodes, labels

def structural_evasion_attack(G, bot_nodes):
    """
    Aggressive Evasion:
    1. Cut ALL ties with other bots.
    2. Connect to high-clustering humans to mimic trust.
    """
    print("⚔️ Applying Structural Evasion Attack (Aggressive)...")
    G_evaded = G.copy()
    
    clustering = nx.clustering(G)
    human_targets = [n for n in G.nodes() if n not in bot_nodes and clustering[n] > 0.5]
    if not human_targets:
        human_targets = [n for n in G.nodes() if n not in bot_nodes]

    for bot in bot_nodes:
        neighbors = list(G_evaded.neighbors(bot))
        for target in neighbors:
            if target in bot_nodes:
                G_evaded.remove_edge(bot, target)
        
       
        num_to_add = 15
        targets = np.random.choice(human_targets, size=num_to_add, replace=False)
        for t in targets:
            if not G_evaded.has_edge(bot, t):
                G_evaded.add_edge(bot, t)
            
    return G_evaded


def graph_poisoning_attack(G, num_poison_nodes=400):
    """
    Injects 400 random-behavior nodes but labels them as HUMAN (0).
    This pollutes the training data.
    """
    print(f"🧪 Applying Graph Poisoning ({num_poison_nodes} nodes)...")
    G_poisoned = G.copy()
    start_id = max(G_poisoned.nodes()) + 1
    
    poison_ids = []
    existing_nodes = list(G.nodes())
    
    for i in range(num_poison_nodes):
        p_id = start_id + i
        poison_ids.append(p_id)
        G_poisoned.add_node(p_id)
        
        targets = random.sample(existing_nodes, 15)
        for t in targets:
            G_poisoned.add_edge(p_id, t)
            
    return G_poisoned, poison_ids


def visualize(G, bot_nodes, title, filename):
    plt.figure(figsize=(10, 8))
    if G.number_of_nodes() > 1500:
        sampled_nodes = random.sample(list(G.nodes()), 1200)
        sampled_nodes = list(set(sampled_nodes) | set(bot_nodes[:100]))
        subG = G.subgraph(sampled_nodes)
    else:
        subG = G

    pos = nx.spring_layout(subG, seed=42, iterations=15)
    
    color_map = []
    for node in subG.nodes():
        if node in bot_nodes:
            color_map.append('red') # Bots
        else:
            color_map.append('blue') # Humans/Poison
            
    nx.draw(subG, pos, node_color=color_map, node_size=20, alpha=0.5, edge_color='#D3D3D3', width=0.5)
    plt.title(title)
    plt.savefig(filename)
    print(f"🖼️ Saved plot: {filename}")
    plt.close()

def main():
    if not os.path.exists("facebook_combined.txt"):
        print("❌ Error: facebook_combined.txt not found!")
        return
        
    G_orig = load_graph("facebook_combined.txt")
    
    G_baseline, bot_nodes, labels_dict = inject_bots(G_orig, fraction=0.05)
    visualize(G_baseline, bot_nodes, "Baseline: Bots (Red) Clustered", "1_baseline.png")
    
    print("\n--- Training Baseline Model ---")
    nodes_b, X_b = extract_features(G_baseline)
    y_b = [labels_dict[n] for n in nodes_b]
    
    X_train, X_test, y_train, y_test = train_test_split(X_b, y_b, test_size=0.3, random_state=42, stratify=y_b)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred)
    print(f"[Baseline Accuracy]: {base_acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n--- executing Structural Evasion ---")
    G_evasion = structural_evasion_attack(G_baseline, bot_nodes)
    
    nodes_e, X_e = extract_features(G_evasion)
    
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_e, y_b, test_size=0.3, random_state=42, stratify=y_b)
    
    y_pred_evasion = clf.predict(X_test_e)
    evasion_acc = accuracy_score(y_test_e, y_pred_evasion)
    
    print(f"[Evasion Accuracy]: {evasion_acc:.4f}")
    print(classification_report(y_test_e, y_pred_evasion, digits=4))
    visualize(G_evasion, bot_nodes, "Evasion: Bots Dispersed", "2_evasion.png")

    print("\n--- executing Graph Poisoning ---")
    G_poisoned, poison_ids = graph_poisoning_attack(G_baseline, num_poison_nodes=400)
    
    labels_poison = labels_dict.copy()
    for pid in poison_ids:
        labels_poison[pid] = 0 
        
    nodes_p, X_p = extract_features(G_poisoned)
    y_p = [labels_poison[n] for n in nodes_p]
    
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.3, random_state=42, stratify=y_p)
    
    clf_poisoned = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_poisoned.fit(X_train_p, y_train_p)
    
    print("\n[Poisoned Model tested on Clean Data]")
    y_pred_poison_transfer = clf_poisoned.predict(X_test) 
    
    poison_acc = accuracy_score(y_test, y_pred_poison_transfer)
    print(f"[Poisoning Accuracy]: {poison_acc:.4f}")
    print(classification_report(y_test, y_pred_poison_transfer, digits=4))
    visualize(G_poisoned, bot_nodes, "Poisoning: Fake Nodes Added", "3_poisoning.png")

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Evasion Accuracy:  {evasion_acc:.4f}  (Drop: {base_acc - evasion_acc:.4f})")
    print(f"Poisoning Accuracy:{poison_acc:.4f}  (Drop: {base_acc - poison_acc:.4f})")
    print("="*30)

if __name__ == "__main__":
    main()

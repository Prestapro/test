"""
HONEST Cross-Space Alignment Tests

Key fix: Different feature/topology views (not identity case)

This tests the REAL question:
- Can alignment work when Agent A and Agent B have DIFFENT feature spaces?
- Can topology matching help when graphs are NOT identical?
"""

import numpy as np
from typing import Tuple, List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.cross_space_alignment import (
    FHRRSpace, AlignmentProtocol, AlignmentResult,
    IdentifiabilityAnalyzer, TransferVerification, AlignmentMethod
)


def generate_different_features(
    n_states: int,
    feature_dim: int,
    overlap: float,
    noise_level: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate HONESTLY different features for two agents.
    
    overlap = amount of shared semantic structure
    noise_level = how much private structure each agent has
    
    Key difference from before:
    - Previously: features_a = features_b (identity case)
    - Now: features_a ≠ features_b with controlled overlap
    """
    np.random.seed(seed)
    
    # Shared semantic component (known to both agents)
    shared = np.random.randn(n_states, feature_dim)
    
    # Agent A's private view (different basis)
    private_a = np.random.randn(n_states, feature_dim)
    basis_a = np.random.randn(feature_dim, feature_dim)
    basis_a, _ = np.linalg.qr(basis_a)  # Random rotation
    
    # Agent B's private view (different basis)
    private_b = np.random.randn(n_states, feature_dim)
    basis_b = np.random.randn(feature_dim, feature_dim)
    basis_b, _ = np.linalg.qr(basis_b)  # Different random rotation
    
    # Combine: overlap controls shared vs private
    features_a = overlap * shared + noise_level * (1 - overlap) * (private_a @ basis_a)
    features_b = overlap * shared + noise_level * (1 - overlap) * (private_b @ basis_b)
    
    # Normalize
    features_a = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-10)
    features_b = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-10)
    
    return features_a, features_b


def generate_different_topologies(
    n_nodes: int,
    edge_prob: float = 0.3,
    similarity: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate HONESTLY different graphs for two agents.
    
    similarity = Jaccard similarity of edge sets
    similarity = 1.0: identical graphs
    similarity = 0.0: completely different graphs
    """
    np.random.seed(seed)
    
    # Shared edges (known to both agents)
    n_shared = int(n_nodes * n_nodes * edge_prob * similarity)
    shared_edges = set()
    while len(shared_edges) < n_shared:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j:
            shared_edges.add((min(i, j), max(i, j)))
    
    # Agent A's private edges
    n_private_a = int(n_nodes * n_nodes * edge_prob * (1 - similarity))
    private_a = set()
    while len(private_a) < n_private_a:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j and (min(i, j), max(i, j)) not in shared_edges:
            private_a.add((min(i, j), max(i, j)))
    
    # Agent B's private edges (different from A's)
    private_b = set()
    while len(private_b) < n_private_a:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j and (min(i, j), max(i, j)) not in shared_edges and (min(i, j), max(i, j)) not in private_a:
            private_b.add((min(i, j), max(i, j)))
    
    # Build adjacency matrices
    adj_a = np.zeros((n_nodes, n_nodes))
    adj_b = np.zeros((n_nodes, n_nodes))
    
    for i, j in shared_edges:
        adj_a[i, j] = adj_a[j, i] = 1
        adj_b[i, j] = adj_b[j, i] = 1
    
    for i, j in private_a:
        adj_a[i, j] = adj_a[j, i] = 1
    
    for i, j in private_b:
        adj_b[i, j] = adj_b[j, i] = 1
    
    return adj_a, adj_b


def honest_feature_alignment_test(
    dimension: int = 8,
    n_states: int = 50,
    overlap_values: List[float] = None,
    n_trials: int = 5
) -> Dict:
    """
    Test feature-based alignment with HONESTLY different feature spaces.
    
    Key question: Can alignment work when features are observed through
    DIFFERENT bases by the two agents?
    """
    if overlap_values is None:
        overlap_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\n" + "=" * 70)
    print("HONEST TEST: Different Feature Views (Not Identity Case)")
    print("=" * 70)
    
    space_a = FHRRSpace.create_random(dimension, name="A", seed=42)
    space_b = FHRRSpace.create_random(dimension, name="B", seed=1042)
    T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
    
    results = {}
    
    for overlap in overlap_values:
        errors = []
        successes = 0
        
        for trial in range(n_trials):
            # Ground truth states
            gt_states = np.random.randn(dimension, n_states) + 1j * np.random.randn(dimension, n_states)
            gt_states = gt_states / np.linalg.norm(gt_states, axis=0, keepdims=True)
            
            states_a = space_a.encode(gt_states)
            states_b = space_b.encode(gt_states)
            
            # HONESTLY different features (not identical!)
            feat_a, feat_b = generate_different_features(
                n_states, dimension, overlap, 
                noise_level=0.5, seed=42 + trial
            )
            
            # Try to align using feature matching
            # The challenge: feat_a ≠ feat_b, only partial overlap
            result = AlignmentProtocol.feature_based_alignment(
                space_a, space_b, feat_a, feat_b, states_a, states_b
            )
            
            errors.append(result.residual_error)
            if result.residual_error < 0.3:
                successes += 1
        
        results[overlap] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'success_rate': successes / n_trials
        }
        
        print(f"  overlap={overlap:.1f}: error={np.mean(errors):.4f}+/-{np.std(errors):.4f}, "
              f"success={successes/n_trials:.2f}")
    
    # Find where alignment starts working
    working_overlap = None
    for overlap in overlap_values:
        if results[overlap]['success_rate'] > 0.5:
            working_overlap = overlap
            break
    
    print(f"\n>>> With HONESTLY different features, alignment works at overlap >= {working_overlap}")
    
    return {
        'results': results,
        'working_overlap': working_overlap,
        'note': 'Features are DIFFERENT between agents (not identity case)'
    }


def honest_topology_alignment_test(
    dimension: int = 8,
    n_states: int = 50,
    similarity_values: List[float] = None,
    n_trials: int = 3
) -> Dict:
    """
    Test topology-based alignment with HONESTLY different graphs.
    
    Key question: Can graph structure help when Agent A and Agent B
    observe DIFFERENT but related networks?
    """
    if similarity_values is None:
        similarity_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    print("\n" + "=" * 70)
    print("HONEST TEST: Different Topology Views (Not Identity Case)")
    print("=" * 70)
    
    space_a = FHRRSpace.create_random(dimension, name="A", seed=42)
    space_b = FHRRSpace.create_random(dimension, name="B", seed=1042)
    T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
    
    results = {}
    
    for similarity in similarity_values:
        errors = []
        successes = 0
        
        for trial in range(n_trials):
            gt_states = np.random.randn(dimension, n_states) + 1j * np.random.randn(dimension, n_states)
            gt_states = gt_states / np.linalg.norm(gt_states, axis=0, keepdims=True)
            
            states_a = space_a.encode(gt_states)
            states_b = space_b.encode(gt_states)
            
            # HONESTLY different topologies (not identical!)
            adj_a, adj_b = generate_different_topologies(
                n_states, edge_prob=0.3, similarity=similarity, seed=42 + trial
            )
            
            result = AlignmentProtocol.topology_based_alignment(
                space_a, space_b, adj_a, adj_b, states_a, states_b
            )
            
            errors.append(result.residual_error)
            if result.residual_error < 0.5:
                successes += 1
        
        results[similarity] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'success_rate': successes / n_trials
        }
        
        print(f"  similarity={similarity:.1f}: error={np.mean(errors):.4f}+/-{np.std(errors):.4f}, "
              f"success={successes/n_trials:.2f}")
    
    working_similarity = None
    for similarity in similarity_values:
        if results[similarity]['success_rate'] > 0.5:
            working_similarity = similarity
            break
    
    print(f"\n>>> With HONESTLY different topologies, alignment works at similarity >= {working_similarity}")
    
    return {
        'results': results,
        'working_similarity': working_similarity,
        'note': 'Graphs are DIFFERENT between agents (not identity case)'
    }


def honest_hybrid_test(
    dimension: int = 8,
    n_states: int = 50,
    k_values: List[int] = None,
    feature_overlap: float = 0.3,
    topology_similarity: float = 0.3,
    n_trials: int = 5
) -> Dict:
    """
    Test hybrid alignment with HONESTLY different features AND topologies.
    
    Key question: Does combining weak signals actually help?
    """
    if k_values is None:
        k_values = [0, 2, 4, 6, 8]
    
    print("\n" + "=" * 70)
    print("HONEST TEST: Hybrid (Anchors + Different Features + Different Topologies)")
    print("=" * 70)
    print(f"Feature overlap: {feature_overlap}, Topology similarity: {topology_similarity}")
    
    space_a = FHRRSpace.create_random(dimension, name="A", seed=42)
    space_b = FHRRSpace.create_random(dimension, name="B", seed=1042)
    T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
    
    results = {}
    
    for k in k_values:
        errors = []
        successes = 0
        
        for trial in range(n_trials):
            gt_states = np.random.randn(dimension, n_states) + 1j * np.random.randn(dimension, n_states)
            gt_states = gt_states / np.linalg.norm(gt_states, axis=0, keepdims=True)
            
            states_a = space_a.encode(gt_states)
            states_b = space_b.encode(gt_states)
            
            # Different features (not identical)
            feat_a, feat_b = generate_different_features(
                n_states, dimension, feature_overlap, seed=42 + trial
            )
            
            # Different topologies (not identical)  
            adj_a, adj_b = generate_different_topologies(
                n_states, similarity=topology_similarity, seed=42 + trial
            )
            
            if k == 0:
                # Pure feature-based (no anchors)
                result = AlignmentProtocol.feature_based_alignment(
                    space_a, space_b, feat_a, feat_b, states_a, states_b
                )
                T_est = result.transform
            elif k >= dimension:
                # Full anchors (trivial solution)
                anchor_a = states_a[:, :k]
                anchor_b = states_b[:, :k]
                result = AlignmentProtocol.anchor_based_alignment(
                    space_a, space_b, anchor_a, anchor_b
                )
                T_est = result.transform
            else:
                # Partial anchors + features + topology
                anchor_a = states_a[:, :k]
                anchor_b = states_b[:, :k]
                
                # Anchor-based initial estimate
                C_anchor = anchor_b @ anchor_a.conj().T
                
                # Feature-based soft correspondence (different features!)
                similarity = feat_a @ feat_b.T
                correspondence = np.exp(similarity) / np.sum(np.exp(similarity), axis=1, keepdims=True)
                states_b_matched = states_b @ correspondence.T
                C_feature = states_b_matched @ states_a.conj().T
                
                # Topology-based constraint (actually use it this time!)
                # Match nodes by graph distance profiles
                try:
                    from scipy.sparse.csgraph import shortest_path
                    dist_a = shortest_path(adj_a, directed=False, unweighted=True)
                    dist_b = shortest_path(adj_b, directed=False, unweighted=True)
                    
                    # Find node correspondences by distance profile similarity
                    n_match = min(n_states, 20)
                    node_scores = np.zeros((n_match, n_match))
                    for i in range(n_match):
                        for j in range(n_match):
                            # Compare distance profiles
                            if np.isfinite(dist_a[i]).any() and np.isfinite(dist_b[j]).any():
                                mask_a = np.isfinite(dist_a[i])
                                mask_b = np.isfinite(dist_b[j])
                                if mask_a.any() and mask_b.any():
                                    profile_a = dist_a[i][mask_a][:n_match]
                                    profile_b = dist_b[j][mask_b][:n_match]
                                    min_len = min(len(profile_a), len(profile_b))
                                    if min_len > 0:
                                        node_scores[i, j] = -np.mean(np.abs(
                                            np.sort(profile_a[:min_len]) - np.sort(profile_b[:min_len])
                                        ))
                    
                    # Greedy matching
                    topo_matches_a = []
                    topo_matches_b = []
                    used_b = set()
                    for i in range(n_match):
                        j = np.argmax(node_scores[i])
                        if j not in used_b and node_scores[i, j] < 0:
                            topo_matches_a.append(i)
                            topo_matches_b.append(j)
                            used_b.add(j)
                    
                    if len(topo_matches_a) >= dimension:
                        topo_states_a = states_a[:, topo_matches_a[:dimension]]
                        topo_states_b = states_b[:, topo_matches_b[:dimension]]
                        C_topo = topo_states_b @ topo_states_a.conj().T
                    else:
                        C_topo = np.zeros_like(C_anchor)
                except:
                    C_topo = np.zeros_like(C_anchor)
                
                # Combine signals
                anchor_weight = k / dimension
                feature_weight = 0.3 * (1 - anchor_weight)
                topo_weight = 0.1 * (1 - anchor_weight)
                remaining = 1.0 - anchor_weight - feature_weight - topo_weight
                
                C = (anchor_weight * C_anchor + 
                     feature_weight * C_feature + 
                     topo_weight * C_topo +
                     remaining * C_anchor)
                
                U, _, Vh = np.linalg.svd(C)
                T_est = U @ Vh
            
            error = np.linalg.norm(T_est - T_true, 'fro') / np.sqrt(dimension)
            errors.append(error)
            
            if error < 0.3:
                successes += 1
        
        results[k] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'success_rate': successes / n_trials
        }
        
        print(f"  k={k}: error={np.mean(errors):.4f}+/-{np.std(errors):.4f}, "
              f"success={successes/n_trials:.2f}")
    
    k_min = None
    for k in k_values:
        if results[k]['success_rate'] > 0.8:
            k_min = k
            break
    
    return {
        'results': results,
        'k_min': k_min,
        'note': 'Features AND topologies are DIFFERENT (honest test)'
    }


def run_honest_tests():
    """Run all HONEST tests with proper different-view setup."""
    print("=" * 70)
    print("HONEST CROSS-SPACE ALIGNMENT TESTS")
    print("(Different views, not identity case)")
    print("=" * 70)
    
    # 1. Feature alignment with different views
    feat_result = honest_feature_alignment_test(n_trials=5)
    
    # 2. Topology alignment with different views
    topo_result = honest_topology_alignment_test(n_trials=3)
    
    # 3. Hybrid with all signals being different
    hybrid_result = honest_hybrid_test(n_trials=5)
    
    print("\n" + "=" * 70)
    print("HONEST SUMMARY")
    print("=" * 70)
    print(f"""
RESULTS WITH DIFFERENT-VIEW SETUP:

1. Feature-based alignment:
   Works at overlap >= {feat_result['working_overlap']}
   (Previously claimed: 0.8, but with identity features)

2. Topology-based alignment:
   Works at similarity >= {topo_result['working_similarity']}
   (Previously: identity topology gave trivial success)

3. Hybrid alignment:
   k_min = {hybrid_result['k_min']} with different features AND topologies
   (Previously: weak features didn't reduce k_min because topology wasn't used)

KEY DIFFERENCES FROM PREVIOUS EXPERIMENTS:
- Features are NOW ACTUALLY DIFFERENT between agents
- Topologies are NOW ACTUALLY DIFFERENT between agents
- Topology signal is NOW ACTUALLY USED in hybrid refinement

CAVEATS STILL APPLY:
- This is still a toy model (random unitary spaces)
- Real-world alignment may need different assumptions
- These results should NOT be generalized without verification
""")


if __name__ == "__main__":
    run_honest_tests()

"""
Cross-Space Alignment in FHRR: Identifiability, Alignment Budget, and Transfer

==============================================================================
FUNDAMENTAL QUESTION:
==============================================================================

What is the MINIMAL SUPERVISION BUDGET required for two random representational
bases to become sufficiently aligned for task-local knowledge transfer?

This decomposes into three formal problems:

1. IDENTIFIABILITY
   - Can we distinguish the correct correspondence from false ones?
   - Depends on: graph symmetry, feature richness, shared vocabulary
   - If the graph is symmetric + features are poor + no shared vocab → UNIDENTIFIABLE

2. ALIGNMENT
   - Even if solution exists, how much info to find it?
   - Anchors, features, constraints, shared topology, shared code/vocabulary
   - Key metric: k_min = minimum anchors for reliable alignment

3. TRANSFER
   - After alignment, verify: is knowledge actually transferred?
   - Or just matching scores improved?

==============================================================================
FORMAL MODEL:
==============================================================================

Two FHRR agents A and B with different internal bases:

    Agent A: State space ψ_A = R_A · φ  (R_A ∈ U(d) random unitary transform)
    Agent B: State space ψ_B = R_B · φ  (R_B ∈ U(d) independent random unitary)

Where φ is some "ground truth" representation.

The alignment problem is to find T_AB such that:
    ψ_A ≈ T_AB · ψ_B

Without supervision, T_AB is gauge-ambiguous (any T_AB · G where G ∈ Gauge group works).

With k anchors (known correspondences), we can estimate T_AB.
Question: What is minimum k for reliable recovery?

==============================================================================
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# MLX compatibility layer
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    class MXCompat:
        """NumPy-based compatibility layer for MLX operations."""
        array = np.array
        zeros = np.zeros
        ones = np.ones
        float32 = np.float32
        Dtype = np.dtype
        
        @staticmethod
        def arctan2(y, x): return np.arctan2(y, x)
        @staticmethod
        def sin(x): return np.sin(x)
        @staticmethod
        def cos(x): return np.cos(x)
        @staticmethod
        def exp(x): return np.exp(x)
        @staticmethod
        def log(x): return np.log(x)
        @staticmethod
        def sqrt(x): return np.sqrt(x)
        @staticmethod
        def stack(arrays, axis=0): return np.stack(arrays, axis=axis)
        @staticmethod
        def clip(x, a_min, a_max): return np.clip(x, a_min, a_max)
        @staticmethod
        def sum(x, axis=None): return np.sum(x, axis=axis)
        @staticmethod
        def mean(x, axis=None): return np.mean(x, axis=axis)
        @staticmethod
        def std(x, axis=None): return np.std(x, axis=axis)
        @staticmethod
        def abs(x): return np.abs(x)
        @staticmethod
        def all(condition): return np.all(condition)
        @staticmethod
        def allclose(a, b, atol=1e-8): return np.allclose(a, b, atol=atol)
        @staticmethod
        def where(condition, x, y): return np.where(condition, x, y)
        @staticmethod
        def ones_like(x): return np.ones_like(x)
        @staticmethod
        def zeros_like(x): return np.zeros_like(x)
        @staticmethod
        def eye(n): return np.eye(n)
        @staticmethod
        def random(): return np.random.random
        @staticmethod
        def matmul(a, b): return np.matmul(a, b)
        @staticmethod
        def transpose(x): return x.T
        @staticmethod
        def conj(x): return np.conj(x)
        @staticmethod
        def real(x): return np.real(x)
        @staticmethod
        def imag(x): return np.imag(x)
        @staticmethod
        def linspace(start, stop, num): return np.linspace(start, stop, num)
        @staticmethod
        def argmax(x, axis=None): return np.argmax(x, axis=axis)
        @staticmethod
        def argsort(x, axis=None): return np.argsort(x, axis=axis)
        @staticmethod
        def max(x, axis=None): return np.max(x, axis=axis)
        @staticmethod
        def min(x, axis=None): return np.min(x, axis=axis)
    
    mx = MXCompat()
    HAS_MLX = False


class AlignmentMethod(Enum):
    """Methods for cross-space alignment."""
    ANCHOR_BASED = "anchor"           # k explicit correspondences
    FEATURE_BASED = "feature"          # Semantic feature matching
    TOPOLOGY_BASED = "topology"        # Graph structure matching
    HYBRID = "hybrid"                  # Combination of methods


@dataclass
class AlignmentResult:
    """Result of an alignment attempt between two FHRR spaces."""
    transform: np.ndarray           # Estimated transformation T_AB
    residual_error: float           # Alignment error
    k_anchors_used: int             # Number of anchors used
    method: AlignmentMethod         # Method used
    identifiability_score: float    # How identifiable the solution is
    transfer_score: float = 0.0     # Task-local transfer quality
    success: bool = False           # Whether alignment succeeded


@dataclass 
class FHRRSpace:
    """
    FHRR Representational Space.
    
    A state space with an internal basis defined by a unitary transformation
    from some ground truth representation.
    """
    dimension: int
    basis_transform: np.ndarray    # R ∈ U(d) - unitary transform from ground truth
    name: str = "Agent"
    
    @classmethod
    def create_random(cls, dimension: int, name: str = "Agent", seed: int = None) -> 'FHRRSpace':
        """Create a random FHRR space with independent basis."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random unitary matrix via QR decomposition
        random_matrix = np.random.randn(dimension, dimension) + 1j * np.random.randn(dimension, dimension)
        Q, R = np.linalg.qr(random_matrix)
        
        # Ensure proper unitary (det = 1 on unit circle)
        D = np.diag(R)
        Q = Q @ np.diag(D / np.abs(D))
        
        return cls(dimension=dimension, basis_transform=Q, name=name)
    
    def encode(self, ground_truth_state: np.ndarray) -> np.ndarray:
        """Encode a ground truth state into this space's basis."""
        return self.basis_transform @ ground_truth_state
    
    def decode(self, internal_state: np.ndarray) -> np.ndarray:
        """Decode from internal basis back to ground truth."""
        return self.basis_transform.conj().T @ internal_state
    
    def get_phase_representation(self, state: np.ndarray) -> np.ndarray:
        """Extract phase angles from complex state."""
        return np.angle(state)


class IdentifiabilityAnalyzer:
    """
    Analyze identifiability of cross-space alignment problem.
    
    Key question: Can we distinguish the correct correspondence from false ones?
    
    Sources of non-identifiability:
    1. Permutation ambiguity: P · T_AB where P is a permutation
    2. Phase ambiguity: diag(e^(iθ)) · T_AB
    3. Gauge freedom: G · T_AB where G preserves structure
    4. Graph symmetry: automorphism group of the graph
    """
    
    @staticmethod
    def compute_spectral_gap(eigenvalues: np.ndarray) -> float:
        """
        Compute spectral gap - relates to uniqueness of alignment.
        
        Larger gap = more identifiable solution.
        """
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
        if len(sorted_eigs) < 2:
            return 0.0
        return float(sorted_eigs[0] - sorted_eigs[1])
    
    @staticmethod
    def compute_symmetry_entropy(basis_transform: np.ndarray, n_samples: int = 100) -> float:
        """
        Estimate symmetry entropy of the basis.
        
        Higher symmetry = lower identifiability.
        We sample random gauge transformations and measure invariance.
        """
        d = basis_transform.shape[0]
        invariance_scores = []
        
        for _ in range(n_samples):
            # Random phase gauge
            phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, d))
            gauge = np.diag(phases)
            
            # Check how much the transformation changes the basis
            transformed = gauge @ basis_transform
            diff = np.linalg.norm(transformed - basis_transform, 'fro')
            invariance_scores.append(np.exp(-diff))
        
        # Higher entropy = more gauge freedom = lower identifiability
        scores = np.array(invariance_scores)
        scores = scores / (scores.sum() + 1e-10)
        entropy = -np.sum(scores * np.log(scores + 1e-10))
        
        return float(entropy)
    
    @staticmethod
    def compute_permutation_ambiguity(
        space_a: FHRRSpace,
        space_b: FHRRSpace,
        n_permutations: int = 100
    ) -> float:
        """
        Measure permutation ambiguity in alignment.
        
        If many permutations give similar alignment scores, 
        the problem has low identifiability.
        """
        d = space_a.dimension
        
        # Compute ground truth transformation
        T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
        
        # Generate random permutations and measure their alignment quality
        alignment_scores = []
        
        for _ in range(n_permutations):
            # Random permutation
            perm = np.random.permutation(d)
            P = np.eye(d)[perm]
            
            # Permuted transformation
            T_perm = P @ T_true
            
            # How different is this from true transformation?
            # Measure: ||T_perm - T_true||_F
            diff = np.linalg.norm(T_perm - T_true, 'fro')
            alignment_scores.append(diff)
        
        # Low variance in differences = high permutation ambiguity
        ambiguity = 1.0 / (np.std(alignment_scores) + 1e-10)
        
        return float(min(ambiguity, 10.0))  # Cap at 10 for numerical stability
    
    @staticmethod
    def analyze(space_a: FHRRSpace, space_b: FHRRSpace) -> Dict[str, float]:
        """
        Full identifiability analysis for cross-space alignment.
        
        Returns dict with:
        - spectral_gap: uniqueness measure
        - symmetry_entropy: gauge freedom
        - permutation_ambiguity: structural ambiguity
        - identifiability_score: composite score (higher = more identifiable)
        """
        # Combine eigenvalues from both spaces
        eigs_a = np.linalg.eigvals(space_a.basis_transform)
        eigs_b = np.linalg.eigvals(space_b.basis_transform)
        
        spectral_gap = (
            IdentifiabilityAnalyzer.compute_spectral_gap(eigs_a) +
            IdentifiabilityAnalyzer.compute_spectral_gap(eigs_b)
        ) / 2
        
        symmetry_entropy = (
            IdentifiabilityAnalyzer.compute_symmetry_entropy(space_a.basis_transform) +
            IdentifiabilityAnalyzer.compute_symmetry_entropy(space_b.basis_transform)
        ) / 2
        
        perm_ambiguity = IdentifiabilityAnalyzer.compute_permutation_ambiguity(space_a, space_b)
        
        # Composite identifiability score
        # Higher = more identifiable
        identifiability = (
            spectral_gap * 10 +           # Larger gap helps
            (1.0 / (symmetry_entropy + 0.1)) +  # Lower entropy helps
            (1.0 / (perm_ambiguity + 0.1))      # Lower ambiguity helps
        )
        
        return {
            'spectral_gap': spectral_gap,
            'symmetry_entropy': symmetry_entropy,
            'permutation_ambiguity': perm_ambiguity,
            'identifiability_score': identifiability
        }


class AlignmentProtocol:
    """
    Protocols for aligning two FHRR spaces with minimal supervision.
    
    Methods:
    1. ANCHOR_BASED: Use k explicit correspondences
    2. FEATURE_BASED: Match semantic features
    3. TOPOLOGY_BASED: Match graph structure
    4. HYBRID: Combine methods
    """
    
    @staticmethod
    def anchor_based_alignment(
        space_a: FHRRSpace,
        space_b: FHRRSpace,
        anchor_states_a: np.ndarray,
        anchor_states_b: np.ndarray,
        regularization: float = 0.01
    ) -> AlignmentResult:
        """
        Align spaces using k anchor correspondences.
        
        Given k anchor pairs (s_a^i, s_b^i), estimate T_AB such that:
            s_b ≈ T_AB · s_a
        
        For FHRR (complex unitary), we use Procrustes alignment:
            T_AB = argmin ||S_B - T · S_A||_F subject to T ∈ U(d)
        
        Solution: T_AB = V · U^H where U·Σ·V^H = SVD(S_B · S_A^H)
        """
        k = anchor_states_a.shape[1]  # Number of anchors
        d = space_a.dimension
        
        # Stack anchor states into matrices
        # S_A = [s_a^1, ..., s_a^k] ∈ C^{d×k}
        # S_B = [s_b^1, ..., s_b^k] ∈ C^{d×k}
        S_A = anchor_states_a  # shape: (d, k)
        S_B = anchor_states_b  # shape: (d, k)
        
        # Compute cross-correlation matrix
        C = S_B @ S_A.conj().T  # C ∈ C^{d×d}
        
        # SVD for Procrustes solution
        U, Sigma, Vh = np.linalg.svd(C)
        
        # Optimal unitary transformation
        T_estimated = U @ Vh
        
        # Compute ground truth for comparison
        T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
        
        # Alignment error (Frobenius norm of difference)
        # Normalize by dimension for fair comparison
        residual_error = np.linalg.norm(T_estimated - T_true, 'fro') / np.sqrt(d)
        
        # Analyze identifiability
        id_analysis = IdentifiabilityAnalyzer.analyze(space_a, space_b)
        
        success = residual_error < 0.5  # Threshold for "successful" alignment
        
        return AlignmentResult(
            transform=T_estimated,
            residual_error=float(residual_error),
            k_anchors_used=k,
            method=AlignmentMethod.ANCHOR_BASED,
            identifiability_score=id_analysis['identifiability_score'],
            success=success
        )
    
    @staticmethod
    def feature_based_alignment(
        space_a: FHRRSpace,
        space_b: FHRRSpace,
        features_a: np.ndarray,
        features_b: np.ndarray,
        states_a: np.ndarray,
        states_b: np.ndarray
    ) -> AlignmentResult:
        """
        Align spaces by matching semantic features.
        
        Use feature similarity to establish soft correspondences,
        then solve for transformation.
        
        Args:
            features_a: Semantic features for states in space A (n_a × f)
            features_b: Semantic features for states in space B (n_b × f)
            states_a: States in space A (d × n_a)
            states_b: States in space B (d × n_b)
        """
        # Compute feature similarity matrix
        # S_ij = similarity(feature_a_i, feature_b_j)
        n_a = features_a.shape[0]
        n_b = features_b.shape[0]
        d = space_a.dimension
        
        # Cosine similarity
        features_a_norm = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-10)
        features_b_norm = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-10)
        
        similarity = features_a_norm @ features_b_norm.T  # (n_a × n_b)
        
        # Soft correspondence via softmax
        correspondence_a = np.exp(similarity * 10) / np.sum(np.exp(similarity * 10), axis=1, keepdims=True)
        
        # Weighted states for alignment
        # Soft-matched states in B
        states_b_matched = states_b @ correspondence_a.T  # (d × n_a)
        
        # Now use anchor-based alignment with soft matches
        C = states_b_matched @ states_a.conj().T
        U, Sigma, Vh = np.linalg.svd(C)
        T_estimated = U @ Vh
        
        T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
        residual_error = np.linalg.norm(T_estimated - T_true, 'fro') / np.sqrt(d)
        
        id_analysis = IdentifiabilityAnalyzer.analyze(space_a, space_b)
        
        return AlignmentResult(
            transform=T_estimated,
            residual_error=float(residual_error),
            k_anchors_used=0,  # No explicit anchors
            method=AlignmentMethod.FEATURE_BASED,
            identifiability_score=id_analysis['identifiability_score'],
            success=residual_error < 0.5
        )
    
    @staticmethod
    def topology_based_alignment(
        space_a: FHRRSpace,
        space_b: FHRRSpace,
        adjacency_a: np.ndarray,
        adjacency_b: np.ndarray,
        states_a: np.ndarray,
        states_b: np.ndarray,
        n_iterations: int = 10
    ) -> AlignmentResult:
        """
        Align spaces by matching graph topology.
        
        Use spectral graph matching to find correspondences,
        then align transformations.
        
        This tests whether TOPOLOGY ALONE is sufficient for alignment.
        """
        d = space_a.dimension
        
        # Spectral embeddings from graph Laplacian
        def get_spectral_embedding(adj):
            D = np.diag(np.sum(adj, axis=1))
            L = D - adj  # Graph Laplacian
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            return eigenvectors[:, :d]  # Use first d eigenvectors
        
        spec_a = get_spectral_embedding(adjacency_a)
        spec_b = get_spectral_embedding(adjacency_b)
        
        # Match spectral embeddings (up to sign ambiguity per dimension)
        # Greedy matching
        n_a = spec_a.shape[0]
        n_b = spec_b.shape[0]
        
        # Compute pairwise distances in spectral space
        distances = np.zeros((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                distances[i, j] = np.linalg.norm(spec_a[i] - spec_b[j])
        
        # Greedy matching
        matches_a = []
        matches_b = []
        used_b = set()
        
        for i in range(min(n_a, n_b)):
            j = np.argmin(distances[i])
            if j not in used_b:
                matches_a.append(i)
                matches_b.append(j)
                used_b.add(j)
        
        if len(matches_a) < d:
            # Not enough matches - alignment fails
            return AlignmentResult(
                transform=np.eye(d, dtype=complex),
                residual_error=float('inf'),
                k_anchors_used=len(matches_a),
                method=AlignmentMethod.TOPOLOGY_BASED,
                identifiability_score=0.0,
                success=False
            )
        
        # Use matched states as anchors
        matched_states_a = states_a[:, matches_a[:d]]
        matched_states_b = states_b[:, matches_b[:d]]
        
        # Procrustes alignment
        C = matched_states_b @ matched_states_a.conj().T
        U, Sigma, Vh = np.linalg.svd(C)
        T_estimated = U @ Vh
        
        T_true = space_b.basis_transform @ space_a.basis_transform.conj().T
        residual_error = np.linalg.norm(T_estimated - T_true, 'fro') / np.sqrt(d)
        
        id_analysis = IdentifiabilityAnalyzer.analyze(space_a, space_b)
        
        return AlignmentResult(
            transform=T_estimated,
            residual_error=float(residual_error),
            k_anchors_used=len(matches_a),
            method=AlignmentMethod.TOPOLOGY_BASED,
            identifiability_score=id_analysis['identifiability_score'],
            success=residual_error < 0.5
        )


class TransferVerification:
    """
    Verify that knowledge transfers after alignment.
    
    Key distinction:
    - Matching: Correspondence is found, but no new knowledge
    - Transfer: New relation/formula can be applied in target space
    
    Test: After alignment, can Agent B correctly apply a relation
    that was learned by Agent A?
    """
    
    @staticmethod
    def verify_relation_transfer(
        space_a: FHRRSpace,
        space_b: FHRRSpace,
        transform_ab: np.ndarray,
        relation_matrix: np.ndarray,
        test_states_a: np.ndarray,
        test_states_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Verify transfer of a relation from A to B.
        
        Agent A knows: y = R · x (relation R)
        After alignment, can Agent B apply R?
        
        Test:
        1. Apply R in space A: y_a = R · x_a
        2. Transform to space B: y_b = T_AB · y_a
        3. Compare with: T_AB · R · T_AB^H · x_b
        """
        d = space_a.dimension
        
        # Relation in space A
        R_a = relation_matrix  # Learned relation in A's basis
        
        # Transfer relation to space B
        R_b = transform_ab @ R_a @ transform_ab.conj().T
        
        # Test on states
        # Ground truth: apply relation in A, then transform
        y_a = R_a @ test_states_a
        y_b_via_a = transform_ab @ y_a
        
        # Transfer: apply transformed relation directly in B
        y_b_direct = R_b @ test_states_b
        
        # Transfer error
        transfer_error = np.linalg.norm(y_b_via_a - y_b_direct, 'fro') / np.linalg.norm(y_b_via_a, 'fro')
        
        # Also check semantic equivalence
        # Are the relations isomorphic?
        relation_similarity = np.abs(np.trace(R_a.conj().T @ R_a) - np.trace(R_b.conj().T @ R_b)) / d
        
        return {
            'transfer_error': float(transfer_error),
            'relation_similarity': float(relation_similarity),
            'transfer_success': transfer_error < 0.1
        }
    
    @staticmethod
    def verify_formula_transfer(
        space_a: FHRRSpace,
        space_b: FHRRSpace,
        transform_ab: np.ndarray,
        formula: Callable[[np.ndarray], np.ndarray],
        test_inputs_a: np.ndarray,
        test_inputs_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Verify transfer of an arbitrary formula.
        
        Formula f: state → state learned in space A
        Can it be applied in space B after alignment?
        """
        # Apply formula in A
        outputs_a = formula(test_inputs_a)
        
        # Transform to B
        outputs_b_via_a = transform_ab @ outputs_a
        
        # Apply transformed formula in B
        # (Formula must be transformed based on alignment)
        # This requires the formula to be "alignment-aware"
        
        # For FHRR, phase-based formulas can be transferred by phase shift
        # f_B(ψ) = T_AB · f_A(T_AB^H · ψ)
        transformed_inputs = transform_ab.conj().T @ test_inputs_b
        transformed_outputs = formula(transformed_inputs)
        outputs_b_direct = transform_ab @ transformed_outputs
        
        transfer_error = np.linalg.norm(outputs_b_via_a - outputs_b_direct, 'fro') / (np.linalg.norm(outputs_b_via_a, 'fro') + 1e-10)
        
        return {
            'transfer_error': float(transfer_error),
            'transfer_success': transfer_error < 0.1
        }


class AlignmentExperiment:
    """
    Full experimental pipeline for measuring:
    1. k_min: minimum anchors for reliable alignment
    2. Transfer quality after alignment
    """
    
    def __init__(self, dimension: int, seed: int = 42):
        """Initialize experiment with two random FHRR spaces."""
        np.random.seed(seed)
        
        self.dimension = dimension
        self.space_a = FHRRSpace.create_random(dimension, name="Agent_A", seed=seed)
        self.space_b = FHRRSpace.create_random(dimension, name="Agent_B", seed=seed + 1000)
        
        # Ground truth transformation
        self.T_true = self.space_b.basis_transform @ self.space_a.basis_transform.conj().T
        
        # Identifiability analysis
        self.identifiability = IdentifiabilityAnalyzer.analyze(self.space_a, self.space_b)
        
        print(f"[Experiment] Dimension: {dimension}")
        print(f"[Experiment] Identifiability Score: {self.identifiability['identifiability_score']:.4f}")
        print(f"[Experiment] Spectral Gap: {self.identifiability['spectral_gap']:.4f}")
        print(f"[Experiment] Symmetry Entropy: {self.identifiability['symmetry_entropy']:.4f}")
        print(f"[Experiment] Permutation Ambiguity: {self.identifiability['permutation_ambiguity']:.4f}")
    
    def generate_anchor_states(self, k: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate k anchor states in both spaces (ground truth correspondence)."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random ground truth states
        gt_states = np.random.randn(self.dimension, k) + 1j * np.random.randn(self.dimension, k)
        gt_states = gt_states / np.linalg.norm(gt_states, axis=0, keepdims=True)  # Normalize
        
        # Encode in each space's basis
        states_a = self.space_a.encode(gt_states)
        states_b = self.space_b.encode(gt_states)
        
        return states_a, states_b
    
    def measure_k_min(
        self,
        k_values: List[int] = None,
        n_trials: int = 10,
        error_threshold: float = 0.3
    ) -> Dict[str, any]:
        """
        Measure minimum k (anchors) for reliable alignment.
        
        For each k, run multiple trials and measure success rate.
        k_min = smallest k with success rate > threshold.
        """
        if k_values is None:
            k_values = list(range(1, self.dimension + 3))
        
        results = {}
        
        for k in k_values:
            successes = 0
            errors = []
            
            for trial in range(n_trials):
                states_a, states_b = self.generate_anchor_states(k, seed=42 + trial * 100)
                
                result = AlignmentProtocol.anchor_based_alignment(
                    self.space_a, self.space_b, states_a, states_b
                )
                
                errors.append(result.residual_error)
                if result.residual_error < error_threshold:
                    successes += 1
            
            success_rate = successes / n_trials
            
            results[k] = {
                'success_rate': success_rate,
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'meets_threshold': success_rate > 0.8
            }
            
            print(f"  k={k}: success_rate={success_rate:.2f}, error={np.mean(errors):.4f}±{np.std(errors):.4f}")
        
        # Find k_min
        k_min = None
        for k in k_values:
            if results[k]['meets_threshold']:
                k_min = k
                break
        
        return {
            'k_min': k_min,
            'results_by_k': results,
            'conclusion': f"Minimum k for reliable alignment: {k_min}" if k_min else "Alignment not achievable with tested k values"
        }
    
    def compare_alignment_methods(self, n_states: int = 50) -> Dict[str, AlignmentResult]:
        """
        Compare different alignment methods.
        
        Test whether topology/features can reduce k_min below d (dimension).
        """
        # Generate states
        gt_states = np.random.randn(self.dimension, n_states) + 1j * np.random.randn(self.dimension, n_states)
        gt_states = gt_states / np.linalg.norm(gt_states, axis=0, keepdims=True)
        
        states_a = self.space_a.encode(gt_states)
        states_b = self.space_b.encode(gt_states)
        
        # Random features (simulating semantic features)
        features = np.random.randn(n_states, self.dimension)
        
        # Random graph topology
        adj = np.random.rand(n_states, n_states)
        adj = (adj + adj.T) / 2  # Symmetric
        adj = (adj > 0.7).astype(float)  # Threshold to create sparse graph
        
        results = {}
        
        # 1. Anchor-based with d anchors (theoretical minimum)
        anchor_a, anchor_b = self.generate_anchor_states(self.dimension)
        results['anchor_d'] = AlignmentProtocol.anchor_based_alignment(
            self.space_a, self.space_b, anchor_a, anchor_b
        )
        
        # 2. Feature-based
        results['feature'] = AlignmentProtocol.feature_based_alignment(
            self.space_a, self.space_b, features, features, states_a, states_b
        )
        
        # 3. Topology-based
        results['topology'] = AlignmentProtocol.topology_based_alignment(
            self.space_a, self.space_b, adj, adj, states_a, states_b
        )
        
        print("\n[Method Comparison]")
        for method, result in results.items():
            print(f"  {method}: error={result.residual_error:.4f}, success={result.success}")
        
        return results
    
    def verify_transfer(
        self,
        transform: np.ndarray,
        n_test: int = 20
    ) -> Dict[str, float]:
        """
        Verify knowledge transfer after alignment.
        
        Test whether a learned relation transfers correctly.
        """
        # Create a random relation (e.g., learned transformation)
        R = np.random.randn(self.dimension, self.dimension) + 1j * np.random.randn(self.dimension, self.dimension)
        R = R / np.linalg.norm(R, 'fro')  # Normalize
        
        # Test states
        gt_test = np.random.randn(self.dimension, n_test) + 1j * np.random.randn(self.dimension, n_test)
        gt_test = gt_test / np.linalg.norm(gt_test, axis=0, keepdims=True)
        
        test_a = self.space_a.encode(gt_test)
        test_b = self.space_b.encode(gt_test)
        
        transfer_result = TransferVerification.verify_relation_transfer(
            self.space_a, self.space_b, transform, R, test_a, test_b
        )
        
        return transfer_result


def run_full_experiment(dimension: int = 8, n_trials: int = 20):
    """
    Run full experimental pipeline.
    
    This answers the core question:
    > What is the minimum supervision budget for cross-space alignment?
    """
    print("=" * 70)
    print("CROSS-SPACE ALIGNMENT IN FHRR: EXPERIMENTAL ANALYSIS")
    print("=" * 70)
    print(f"\nCore Question: What is k_min for reliable alignment?")
    print(f"Dimension: {dimension}")
    print()
    
    # Initialize experiment
    exp = AlignmentExperiment(dimension=dimension, seed=42)
    
    # 1. Measure k_min
    print("\n[1] MEASURING k_min (Minimum Anchors for Alignment)")
    print("-" * 50)
    k_min_result = exp.measure_k_min(n_trials=n_trials)
    print(f"\n>>> RESULT: {k_min_result['conclusion']}")
    
    # 2. Compare alignment methods
    print("\n[2] COMPARING ALIGNMENT METHODS")
    print("-" * 50)
    method_results = exp.compare_alignment_methods()
    
    # 3. Verify transfer
    print("\n[3] VERIFYING KNOWLEDGE TRANSFER")
    print("-" * 50)
    
    # Use the best alignment for transfer test
    best_method = min(method_results.items(), key=lambda x: x[1].residual_error)
    print(f"Best method: {best_method[0]} with error {best_method[1].residual_error:.4f}")
    
    if best_method[1].residual_error < 0.5:
        transfer = exp.verify_transfer(best_method[1].transform)
        print(f"Transfer error: {transfer['transfer_error']:.4f}")
        print(f"Transfer success: {transfer['transfer_success']}")
    else:
        print("Alignment error too high for meaningful transfer test")
        transfer = {'transfer_error': float('inf'), 'transfer_success': False}
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 70)
    print(f"""
    IDENTIFIABILITY ANALYSIS:
    - Spectral Gap: {exp.identifiability['spectral_gap']:.4f}
    - Symmetry Entropy: {exp.identifiability['symmetry_entropy']:.4f}
    - Permutation Ambiguity: {exp.identifiability['permutation_ambiguity']:.4f}
    - Composite Score: {exp.identifiability['identifiability_score']:.4f}
    
    ALIGNMENT RESULTS:
    - k_min (anchor-based): {k_min_result['k_min']}
    - Best method: {best_method[0]}
    - Best error: {best_method[1].residual_error:.4f}
    
    TRANSFER VERIFICATION:
    - Transfer error: {transfer['transfer_error']:.4f}
    - Transfer success: {transfer['transfer_success']}
    
    CORE FINDING:
    """)
    
    if k_min_result['k_min'] is not None:
        print(f"    >>> k_min = {k_min_result['k_min']} anchors needed for dimension d = {dimension}")
        print(f"    >>> Ratio k_min/d = {k_min_result['k_min']/dimension:.2f}")
    else:
        print(f"    >>> Alignment not achievable with k ≤ {dimension + 2}")
    
    return {
        'k_min': k_min_result['k_min'],
        'identifiability': exp.identifiability,
        'method_results': {k: v.residual_error for k, v in method_results.items()},
        'transfer': transfer
    }


if __name__ == "__main__":
    run_full_experiment(dimension=8, n_trials=15)

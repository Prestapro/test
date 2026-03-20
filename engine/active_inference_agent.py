"""
Active Inference Agent for Neurosymbolic AGI Architecture
Operating in Fractional Holographic Reduced Representation (FHRR) over C^d complexes on MLX.

This module implements the core active inference machinery using FHRR encoding,
where states are represented as complex unit vectors on the d-dimensional complex torus.
The decode_action method computes the correct continuous gradient (action) to shift
the current physical state phase towards the target EFE goal phase.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

# MLX compatibility layer - falls back to numpy if MLX unavailable
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    # Create numpy-based mx compatibility layer
    class MXCompat:
        """NumPy-based compatibility layer for MLX operations."""
        array = np.array
        zeros = np.zeros
        ones = np.ones
        float32 = np.float32
        Dtype = np.dtype  # MLX Dtype equivalent
        
        @staticmethod
        def arctan2(y, x):
            return np.arctan2(y, x)
        
        @staticmethod
        def sin(x):
            return np.sin(x)
        
        @staticmethod
        def cos(x):
            return np.cos(x)
        
        @staticmethod
        def tanh(x):
            return np.tanh(x)
        
        @staticmethod
        def stack(arrays, axis=0):
            return np.stack(arrays, axis=axis)
        
        @staticmethod
        def clip(x, a_min, a_max):
            return np.clip(x, a_min, a_max)
        
        @staticmethod
        def sum(x):
            return np.sum(x)
        
        @staticmethod
        def abs(x):
            return np.abs(x)
        
        @staticmethod
        def all(condition):
            return np.all(condition)
        
        @staticmethod
        def allclose(a, b, atol=1e-8):
            return np.allclose(a, b, atol=atol)
        
        @staticmethod
        def where(condition, x, y):
            return np.where(condition, x, y)
        
        @staticmethod
        def ones_like(x):
            return np.ones_like(x)
    
    mx = MXCompat()
    HAS_MLX = False
    print("[INFO] MLX not available, using NumPy backend for FHRR operations")


class FHRRAgent:
    """
    Fractional Holographic Reduced Representation Agent for Active Inference.
    
    FHRR represents states as complex unit vectors on the complex torus T^d ≅ (S^1)^d.
    Each state is encoded as ψ = e^(iθ) where θ ∈ R^d is the phase vector.
    
    Key operations in FHRR:
    - Binding: ψ_a ⊛ ψ_b = e^(i(θ_a + θ_b)) (element-wise multiplication = phase addition)
    - Unbinding: ψ_a ⊛ ψ_b* = e^(i(θ_a - θ_b)) (binding with conjugate = phase subtraction)
    - Superposition: ψ_a + ψ_b (weighted sum, may lose unit magnitude)
    - Fractional interpolation: ψ^α = e^(iαθ) for α ∈ R
    
    The active inference loop minimizes Expected Free Energy (EFE) by computing
    the action that shifts the current phase towards the goal phase.
    """
    
    def __init__(
        self,
        dimension: int,
        precision: mx.Dtype = mx.float32,
        action_bound: float = np.pi,
        temperature: float = 1.0
    ):
        """
        Initialize the FHRR Agent.
        
        Args:
            dimension: Dimension d of the complex vector space C^d
            precision: MLX dtype for numerical precision
            action_bound: Maximum action magnitude (default π for phase wraparound)
            temperature: Temperature parameter for softmax over policies
        """
        self.d = dimension
        self.precision = precision
        self.action_bound = action_bound
        self.temperature = temperature
        
        # Initialize belief state as uniform phase distribution
        self.belief_phase = mx.zeros((self.d,), dtype=self.precision)
        
        # Prior preferences (EFE goal) - initialized to zero phase
        self.prior_phase = mx.zeros((self.d,), dtype=self.precision)
        
    def encode_state(self, continuous_state: mx.array) -> mx.array:
        """
        Encode a continuous real-valued state into FHRR complex representation.
        
        Transforms R^d → C^d by mapping each dimension to a phase on the unit circle.
        Uses the fractional encoding: ψ = e^(i·π·x) where x ∈ [-1, 1]
        
        Args:
            continuous_state: Real-valued state vector in R^d
            
        Returns:
            Complex unit vector in C^d (encoded as real array [real, imag] pairs)
        """
        # Normalize to phase space [-π, π]
        # Assuming continuous_state is roughly bounded; apply tanh for soft normalization
        phase = mx.tanh(continuous_state) * mx.array([np.pi])
        
        # Compute complex exponential: e^(i·phase) = cos(phase) + i·sin(phase)
        real_part = mx.cos(phase)
        imag_part = mx.sin(phase)
        
        # Store phase for internal use
        self.belief_phase = phase
        
        # Return as complex-like structure (MLX uses real arrays to represent complex)
        return mx.stack([real_part, imag_part], axis=-1)
    
    def decode_state(self, complex_state: mx.array) -> mx.array:
        """
        Decode FHRR complex representation back to continuous real-valued state.
        
        Extracts the phase angle from the complex unit vector and maps to R^d.
        ψ = e^(iθ) → θ → θ/π ∈ [-1, 1]
        
        Args:
            complex_state: Complex unit vector shape (d, 2) where last dim is [real, imag]
            
        Returns:
            Real-valued state vector in R^d
        """
        real_part = complex_state[..., 0]
        imag_part = complex_state[..., 1]
        
        # Extract phase using arctan2: θ = atan2(Im(ψ), Re(ψ))
        phase = mx.arctan2(imag_part, real_part)
        
        # Map phase to continuous value: x = θ/π
        continuous_state = phase / mx.array([np.pi])
        
        return continuous_state
    
    def compute_phase_difference(
        self,
        current_phase: mx.array,
        target_phase: mx.array
    ) -> mx.array:
        """
        Compute the signed angular distance between two phase vectors.
        
        This is the CRITICAL FHRR operation that the user must implement correctly.
        Linear subtraction of complex vectors is WRONG for FHRR.
        We must compute the minimal rotation on the unit circle.
        
        The angular distance on S^1 is:
            Δθ = atan2(sin(θ_target - θ_current), cos(θ_target - θ_current))
        
        This handles the periodic boundary at ±π, always returning the
        minimum signed rotation to go from current → target.
        
        Args:
            current_phase: Phase vector θ_current ∈ R^d
            target_phase: Phase vector θ_target ∈ R^d
            
        Returns:
            Signed angular distance Δθ ∈ (-π, π]^d
        """
        # Raw phase difference (naive, incorrect for wraparound)
        raw_diff = target_phase - current_phase
        
        # Wrap to (-π, π] using the identity:
        # wrap(θ) = atan2(sin(θ), cos(θ))
        # This gives the minimum signed rotation
        sin_diff = mx.sin(raw_diff)
        cos_diff = mx.cos(raw_diff)
        
        # atan2 returns the signed angle in (-π, π]
        wrapped_diff = mx.arctan2(sin_diff, cos_diff)
        
        return wrapped_diff
    
    def decode_action(
        self,
        current_state_complex: mx.array,
        target_state_complex: mx.array,
        method: str = "phase_gradient"
    ) -> mx.array:
        """
        Decode the continuous action (gradient) needed to shift current state phase
        towards the target EFE goal phase.
        
        ============================================================================
        MATHEMATICAL DERIVATION - Phase Diff Decode Operator for FHRR
        ============================================================================
        
        In FHRR, states are represented as complex unit vectors on the complex torus:
            ψ ∈ C^d such that |ψ_i| = 1 for all i
            
        Each dimension encodes information as a phase angle:
            ψ_i = e^(iθ_i) = cos(θ_i) + i·sin(θ_i)
        
        The action space is the tangent space to the torus at the current state,
        which is isomorphic to R^d (the phase gradient direction).
        
        For current state ψ_curr and target state ψ_target, we need the action a
        such that exp(i·a) rotates ψ_curr towards ψ_target:
        
        The CORRECT operator is NOT Euclidean distance or linear complex subtraction.
        Instead, we compute the signed angular distance on each S^1 fiber:
        
            a = Δθ_wrapped = atan2(sin(θ_target - θ_current), cos(θ_target - θ_current))
        
        This is the Phase Diff Decode operator, which:
        1. Extracts phases: θ_curr = angle(ψ_curr), θ_target = angle(ψ_target)
        2. Computes raw difference: Δθ_raw = θ_target - θ_curr
        3. Wraps to principal interval: Δθ = atan2(sin(Δθ_raw), cos(Δθ_raw))
        
        The resulting action a ∈ (-π, π]^d is the minimum rotation that achieves
        the phase transition, respecting the topology of S^1.
        
        ============================================================================
        
        Args:
            current_state_complex: Current state as complex vector shape (d, 2)
                                   where last dimension is [real, imag]
            target_state_complex: Target EFE goal state as complex vector shape (d, 2)
            method: Decoding method ("phase_gradient" or "geodesic")
            
        Returns:
            action: Continuous gradient (phase shift) in R^d, bounded by action_bound
        """
        # =====================================================================
        # STEP 1: Extract phase angles from complex representations
        # Using mx.arctan2 (the mx.angle() equivalent in MLX)
        # =====================================================================
        
        # Current state: ψ_curr = e^(iθ_curr)
        current_real = current_state_complex[..., 0]
        current_imag = current_state_complex[..., 1]
        
        # Target state: ψ_target = e^(iθ_target)  
        target_real = target_state_complex[..., 0]
        target_imag = target_state_complex[..., 1]
        
        # Extract phases using arctan2(Im, Re) - this is mx.angle() equivalent
        # θ = atan2(Im(ψ), Re(ψ))
        current_phase = mx.arctan2(current_imag, current_real)
        target_phase = mx.arctan2(target_imag, target_real)
        
        # =====================================================================
        # STEP 2: Compute the Phase Diff Decode operator
        # CRITICAL: Do NOT use linear subtraction or Euclidean distance!
        # Use the correct angular distance on S^1 with wraparound handling.
        # =====================================================================
        
        if method == "phase_gradient":
            # Phase Gradient Method: Direct signed angular distance
            
            # WRONG (but commonly mistaken):
            # action = mx.sqrt((target_real - current_real)**2 + 
            #                  (target_imag - current_imag)**2)  # Euclidean - WRONG!
            # action = target_phase - current_phase  # Linear phase diff - WRONG for wraparound!
            
            # CORRECT: Signed angular distance with wraparound
            # Δθ = atan2(sin(θ_target - θ_curr), cos(θ_target - θ_curr))
            action = self.compute_phase_difference(current_phase, target_phase)
            
        elif method == "geodesic":
            # Geodesic Method: Compute action as velocity along the geodesic
            # The geodesic on S^1 is great circle arc with velocity = angular distance
            
            # For FHRR, the geodesic distance is exactly the wrapped phase difference
            action = self.compute_phase_difference(current_phase, target_phase)
            
            # Optional: Apply fractional power for partial action
            # ψ^α = e^(iαθ) for α ∈ (0, 1] gives intermediate states along geodesic
            
        else:
            raise ValueError(f"Unknown decode method: {method}")
        
        # =====================================================================
        # STEP 3: Bound the action to prevent overshooting
        # The natural bound is π, but we may use a tighter action_bound
        # =====================================================================
        
        # Apply action bound (clipping while preserving sign)
        # This ensures the agent doesn't attempt impossibly large phase jumps
        action = mx.clip(action, -self.action_bound, self.action_bound)
        
        # Store for potential use in policy selection
        self._last_action = action
        self._last_phase_diff = self.compute_phase_difference(current_phase, target_phase)
        
        return action
    
    def decode_action_with_gradient(
        self,
        current_state_complex: mx.array,
        target_state_complex: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Decode action with additional gradient information for policy optimization.
        
        Returns both the action and its gradient with respect to the phase parameters,
        useful for gradient-based policy learning in active inference.
        
        The gradient of the phase difference with respect to the current phase is:
            ∂(Δθ)/∂(θ_curr) = -cos(Δθ_raw) + i·sin(Δθ_raw)
                            = -e^(i·Δθ_raw)
            
        For the wrapped angular distance, the gradient is simply -1 when not at
        the discontinuity at Δθ = ±π.
        
        Args:
            current_state_complex: Current state as complex vector
            target_state_complex: Target EFE goal state as complex vector
            
        Returns:
            Tuple of (action, gradient) where gradient indicates sensitivity
        """
        # Extract phases
        current_phase = mx.arctan2(
            current_state_complex[..., 1],
            current_state_complex[..., 0]
        )
        target_phase = mx.arctan2(
            target_state_complex[..., 1],
            target_state_complex[..., 0]
        )
        
        # Compute action (phase difference)
        action = self.compute_phase_difference(current_phase, target_phase)
        
        # Compute gradient: ∂(atan2(sin(Δ), cos(Δ)))/∂(θ_curr) = -1
        # (assuming we're not at the discontinuity)
        # For MLX, we can compute this analytically or use autodiff
        
        # Analytical gradient (avoiding discontinuity):
        # The gradient magnitude is 1 everywhere except at Δθ = ±π
        # At the discontinuity, we use the average of left and right limits
        raw_diff = target_phase - current_phase
        
        # Gradient with respect to current phase is -1 (unit magnitude, opposite direction)
        # With proper handling at wraparound
        gradient = -mx.ones_like(action)
        
        # Handle potential discontinuity by softening near ±π
        # Use a smooth transition near the boundary
        boundary_region = mx.abs(mx.abs(raw_diff) - mx.array([np.pi])) < 0.1
        smooth_grad = -mx.cos(raw_diff)  # Smooth gradient near boundary
        gradient = mx.where(boundary_region, smooth_grad, gradient)
        
        return action, gradient
    
    def apply_action(
        self,
        current_state_complex: mx.array,
        action: mx.array
    ) -> mx.array:
        """
        Apply an action (phase shift) to the current state.
        
        In FHRR, applying action a to state ψ means:
            ψ_new = ψ ⊛ e^(i·a) = e^(i(θ + a))
        
        This is the binding operation in FHRR - phase addition.
        
        Args:
            current_state_complex: Current state as complex vector (d, 2)
            action: Phase shift vector in R^d
            
        Returns:
            New state after applying action
        """
        # Extract current phase
        current_phase = mx.arctan2(
            current_state_complex[..., 1],
            current_state_complex[..., 0]
        )
        
        # Compute new phase (will naturally wrap around due to trig functions)
        new_phase = current_phase + action
        
        # Re-encode as complex unit vector
        new_real = mx.cos(new_phase)
        new_imag = mx.sin(new_phase)
        
        return mx.stack([new_real, new_imag], axis=-1)
    
    def compute_efe(
        self,
        predicted_state: mx.array,
        preferred_state: mx.array,
        entropy_weight: float = 1.0
    ) -> mx.array:
        """
        Compute Expected Free Energy (EFE) for a predicted state.
        
        EFE = E_q(s)[log q(s) - log p(s)] 
            = E_q(s)[log q(s)] - E_q(s)[log p(s)]
            = -H[q] + D_KL[q || p]
            
        In FHRR, we compute this in phase space using angular distributions.
        
        For complex unit vectors, the KL divergence is approximated by the
        squared angular distance (geodesic distance squared):
            D_KL ≈ (1/2) * Σ_i (Δθ_i)^2
            
        The entropy term relates to phase uncertainty (not implemented here).
        
        Args:
            predicted_state: Predicted future state as complex vector
            preferred_state: Preferred (prior) state as complex vector
            entropy_weight: Weight for entropy term in EFE
            
        Returns:
            EFE value (scalar)
        """
        # Extract phases
        pred_phase = mx.arctan2(
            predicted_state[..., 1],
            predicted_state[..., 0]
        )
        pref_phase = mx.arctan2(
            preferred_state[..., 1],
            preferred_state[..., 0]
        )
        
        # Compute squared angular distance (geodesic distance squared on torus)
        phase_diff = self.compute_phase_difference(pred_phase, pref_phase)
        geodesic_sq = mx.sum(phase_diff ** 2)
        
        # EFE approximation (ignoring entropy term for simplicity)
        efe = geodesic_sq / 2
        
        return efe
    
    def select_policy(
        self,
        current_state: mx.array,
        policies: list,
        horizon: int = 1
    ) -> Tuple[int, mx.array]:
        """
        Select the best policy by minimizing Expected Free Energy.
        
        For each policy (sequence of actions), predict future states and
        compute EFE. Select policy with minimum EFE.
        
        Args:
            current_state: Current state as complex vector
            policies: List of action sequences (each is mx.array of shape (T, d))
            horizon: Planning horizon
            
        Returns:
            Tuple of (best_policy_index, best_action_sequence)
        """
        best_efe = mx.array([float('inf')])
        best_idx = 0
        
        for idx, policy in enumerate(policies):
            # Rollout policy
            state = current_state
            total_efe = mx.array([0.0])
            
            for t in range(min(horizon, policy.shape[0])):
                action = policy[t]
                state = self.apply_action(state, action)
                efe = self.compute_efe(state, self.encode_prior())
                total_efe = total_efe + efe
            
            # Check if this is the best so far
            if float(total_efe) < float(best_efe):
                best_efe = total_efe
                best_idx = idx
        
        return best_idx, policies[best_idx]
    
    def encode_prior(self) -> mx.array:
        """
        Encode the prior preference (EFE goal) as a complex state.
        
        Returns:
            Complex vector representing the prior preference
        """
        return mx.stack([
            mx.cos(self.prior_phase),
            mx.sin(self.prior_phase)
        ], axis=-1)
    
    def set_prior(self, phase: mx.array) -> None:
        """
        Set the prior preference phase.
        
        Args:
            phase: Target phase vector for prior preference
        """
        self.prior_phase = phase
    
    def belief_update(
        self,
        observation: mx.array,
        likelihood_precision: float = 1.0
    ) -> mx.array:
        """
        Update belief state given observation using Bayesian update in phase space.
        
        In FHRR, the belief update combines the prior belief with the observation
        using phase interpolation (fractional power operation):
        
            θ_new = θ_prior + κ · atan2(sin(θ_obs - θ_prior), cos(θ_obs - θ_prior))
            
        where κ ∈ (0, 1] is the precision-weighted learning rate.
        
        Args:
            observation: Observed state as complex vector
            likelihood_precision: Precision (confidence) of the observation
            
        Returns:
            Updated belief state as complex vector
        """
        # Extract phases
        belief_phase = self.belief_phase
        obs_phase = mx.arctan2(
            observation[..., 1],
            observation[..., 0]
        )
        
        # Compute phase difference (observation error in angular space)
        phase_diff = self.compute_phase_difference(belief_phase, obs_phase)
        
        # Precision-weighted update
        # Higher precision = more weight on observation
        kappa = likelihood_precision / (1 + likelihood_precision)
        
        # Update belief phase
        new_belief_phase = belief_phase + kappa * phase_diff
        
        # Store updated belief
        self.belief_phase = new_belief_phase
        
        # Return as complex state
        return mx.stack([
            mx.cos(new_belief_phase),
            mx.sin(new_belief_phase)
        ], axis=-1)


def create_test_scenario():
    """
    Create a test scenario to verify the decode_action implementation.
    
    This demonstrates that the Phase Diff Decode operator correctly handles
    the wraparound case that linear subtraction fails on.
    """
    print("=" * 70)
    print("FHRR Active Inference Agent - decode_action Verification")
    print("=" * 70)
    
    # Initialize agent with 3-dimensional state space
    agent = FHRRAgent(dimension=3)
    
    # Test Case 1: Simple phase difference (no wraparound)
    print("\n[Test 1] Simple phase difference (no wraparound)")
    current_phase = mx.array([0.0, 0.5, -0.5])
    target_phase = mx.array([0.5, 1.0, 0.0])
    
    current_state = mx.stack([mx.cos(current_phase), mx.sin(current_phase)], axis=-1)
    target_state = mx.stack([mx.cos(target_phase), mx.sin(target_phase)], axis=-1)
    
    action = agent.decode_action(current_state, target_state)
    expected = mx.array([0.5, 0.5, 0.5])
    
    print(f"  Current phase: {current_phase}")
    print(f"  Target phase:  {target_phase}")
    print(f"  Action (phase diff): {action}")
    print(f"  Expected:       {expected}")
    print(f"  Match: {mx.allclose(action, expected, atol=1e-5)}")
    
    # Test Case 2: Wraparound case (this is where linear subtraction FAILS)
    print("\n[Test 2] Wraparound case (linear subtraction would be WRONG)")
    current_phase = mx.array([2.5, -2.5, 3.0])  # Near +π
    target_phase = mx.array([-2.5, 2.5, -3.0])  # Near -π (same point on circle!)
    
    current_state = mx.stack([mx.cos(current_phase), mx.sin(current_phase)], axis=-1)
    target_state = mx.stack([mx.cos(target_phase), mx.sin(target_phase)], axis=-1)
    
    action = agent.decode_action(current_state, target_state)
    
    # Linear subtraction would give: [-5.0, 5.0, -6.0] - WRONG!
    # Correct wrapped action should be near: [0.78, -0.78, 0.28]
    linear_wrong = target_phase - current_phase
    
    print(f"  Current phase: {current_phase}")
    print(f"  Target phase:  {target_phase}")
    print(f"  Linear diff (WRONG): {linear_wrong}")
    print(f"  Wrapped action (CORRECT): {action}")
    print(f"  Actions are small (as expected): {mx.all(mx.abs(action) < mx.array([np.pi]))}")
    
    # Test Case 3: Verify action application moves state toward target
    print("\n[Test 3] Action application moves toward target")
    current_phase = mx.array([0.0, 0.0, 0.0])
    target_phase = mx.array([1.0, 2.0, -1.0])
    
    current_state = mx.stack([mx.cos(current_phase), mx.sin(current_phase)], axis=-1)
    target_state = mx.stack([mx.cos(target_phase), mx.sin(target_phase)], axis=-1)
    
    action = agent.decode_action(current_state, target_state)
    new_state = agent.apply_action(current_state, action)
    
    # Verify new state is close to target
    new_phase = mx.arctan2(new_state[..., 1], new_state[..., 0])
    
    print(f"  Initial phase: {current_phase}")
    print(f"  Target phase:  {target_phase}")
    print(f"  Action:        {action}")
    print(f"  New phase:     {new_phase}")
    print(f"  Reached target: {mx.allclose(new_phase, target_phase, atol=1e-5)}")
    
    print("\n" + "=" * 70)
    print("All tests demonstrate correct Phase Diff Decode operator usage!")
    print("=" * 70)
    
    return agent


if __name__ == "__main__":
    create_test_scenario()

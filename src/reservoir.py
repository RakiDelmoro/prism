"""
E-I Reservoir with Emergent Clusters and Generalized Synchronization Readout

Based on:
- arXiv 2504.12480: "Boosting Reservoir Computing with Brain-inspired Adaptive Dynamics"
- arXiv 2411.10991: "Modulating Reservoir Dynamics via RL" (DARC)
- Nature Sci. Rep.: "Reservoir computing with generalized readout based on generalized synchronization"

Architecture:
  Game Features [8] → E-I Reservoir [500] → Cluster Means → Argmax → Action
                            ↑                    ↓
                     inhibitory plasticity    emergent clusters
                     (self-organization)      (correlation analysis)

Action selection uses generalized synchronization:
  - Reservoir synchronizes to different dynamical modes based on input
  - Each emergent cluster represents one synchronized mode
  - Action = which mode the reservoir is currently synchronized to

No readout weights, no PPO, no context policy.
Pure unsupervised self-organization + emergent cluster discovery.
"""
import torch
import torch.nn as nn
import numpy as np


class EIReservoir(nn.Module):
    """
    Excitatory-Inhibitory Reservoir with Emergent Clusters.

    Self-organizes via inhibitory plasticity (Eq. 7) to achieve criticality.
    Clusters emerge from correlation analysis after self-organization.
    Action selection via cluster competition (generalized synchronization readout).
    """

    def __init__(
        self,
        input_dim: int = 8,
        n_neurons: int = 500,
        n_actions: int = 3,
        f_exc: float = 0.8,
        degree: int = 50,
        sigmoid_c: float = 10.0,
        threshold: float = 0.0,
        leakage: float = 0.0,
        f_in: float = 0.3,
        sigma_in: float = 0.5,
        plasticity_delta: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_exc = int(n_neurons * f_exc)
        self.n_inh = n_neurons - self.n_exc
        self.n_actions = n_actions
        self.degree = degree
        self.sigmoid_c = sigmoid_c
        self.threshold = threshold
        self.leakage = leakage
        self.f_in = f_in
        self.sigma_in = sigma_in
        self.plasticity_delta = plasticity_delta
        self.device = device
        self.input_dim = input_dim

        # Build connectivity (Dale's Law: E→all, I→all, non-negative weights)
        self._build_connectivity()

        # Build input weights
        self._build_input_weights()

        # Target rates ~ Beta(2,2) — heterogeneous for cluster formation
        rho = np.random.beta(2, 2, size=n_neurons).astype(np.float32)
        self.register_buffer("rho", torch.tensor(rho, device=device))

        # State
        self.register_buffer("V", torch.zeros(1, n_neurons, device=device))
        self.register_buffer("r", torch.zeros(1, n_neurons, device=device))

        # Emergent cluster membership (filled by discover_clusters)
        # Shape: [n_neurons, n_actions]
        self.register_buffer("cluster_membership", torch.eye(n_neurons, n_actions, device=device))

        self.frozen = False

    # ── Construction ──────────────────────────────────────────────────

    def _build_connectivity(self):
        """Build sparse E and I connectivity with Dale's Law (paper Table I)."""
        n = self.n_neurons
        n_e = self.n_exc
        n_i = self.n_inh
        k = self.degree
        f_e = n_e / n

        mu_E = 1.0 / (k * f_e)
        sigma_E = 0.2 * mu_E

        f_i = n_i / n
        mu_I = (f_e / f_i) * mu_E
        sigma_I = 0.2 * mu_E

        rows_e, cols_e, vals_e = [], [], []
        for post in range(n):
            pres = np.random.choice(n_e, size=k, replace=False)
            for pre in pres:
                w = np.abs(np.random.normal(mu_E, sigma_E))
                rows_e.append(pre)
                cols_e.append(post)
                vals_e.append(w)

        rows_i, cols_i, vals_i = [], [], []
        for post in range(n):
            pres = np.random.choice(n_i, size=k, replace=False) + n_e
            for pre in pres:
                w = np.abs(np.random.normal(mu_I, sigma_I))
                rows_i.append(pre)
                cols_i.append(post)
                vals_i.append(w)

        self.register_buffer("A_E_indices",
            torch.tensor([rows_e, cols_e], dtype=torch.long, device=self.device))
        self.register_buffer("A_E_values",
            torch.tensor(vals_e, dtype=torch.float32, device=self.device))
        self.register_buffer("A_I_indices",
            torch.tensor([rows_i, cols_i], dtype=torch.long, device=self.device))
        self.register_buffer("A_I_values",
            torch.tensor(vals_i, dtype=torch.float32, device=self.device))

    def _build_input_weights(self):
        """W_in: [N, input_dim]. Only f_in fraction receive input (random)."""
        n_input = int(self.n_neurons * self.f_in)
        input_ids = np.random.choice(self.n_neurons, size=n_input, replace=False)

        W_in = torch.zeros(self.n_neurons, self.input_dim, device=self.device)
        W_in[input_ids] = torch.zeros(len(input_ids), self.input_dim, device=self.device).uniform_(
            -self.sigma_in / 2, self.sigma_in / 2
        )
        self.register_buffer("W_in", W_in)

    # ── Dynamics ──────────────────────────────────────────────────────

    def reset_state(self, batch_size: int = 1):
        """Reset reservoir state."""
        self.V = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.r = torch.zeros(batch_size, self.n_neurons, device=self.device)

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        """
        One timestep dynamics (paper Eq. 1):
          V_{t+1} = (A^E − A^I)·r_t + W_in·u_t
          r_{t+1} = σ(c·(V_{t+1} − θ))
        """
        W_in_u = torch.matmul(u_t, self.W_in.T)

        A_E = torch.sparse_coo_tensor(
            self.A_E_indices, self.A_E_values,
            (self.n_neurons, self.n_neurons)
        ).coalesce()
        A_I = torch.sparse_coo_tensor(
            self.A_I_indices, self.A_I_values,
            (self.n_neurons, self.n_neurons)
        ).coalesce()

        E_contrib = torch.sparse.mm(A_E.t(), self.r.t()).t()
        I_contrib = torch.sparse.mm(A_I.t(), self.r.t()).t()

        self.V = self.leakage * self.V + E_contrib - I_contrib + W_in_u
        self.r = torch.sigmoid(self.sigmoid_c * (self.V - self.threshold))

        return self.r

    # ── Self-organization (paper Eq. 7 & 8) ───────────────────────────

    def one_step_design(self, mean_input: torch.Tensor):
        """One-step design (Eq. 8). Sets initial inhibitory weights."""
        with torch.no_grad():
            rho = self.rho
            logit_rho = torch.log(rho / (1 - rho + 1e-8) + 1e-8)

            w_in_u = self.W_in @ mean_input

            A_E = torch.sparse_coo_tensor(
                self.A_E_indices, self.A_E_values,
                (self.n_neurons, self.n_neurons)
            ).coalesce()
            exc_drive = A_E @ rho

            A_I = torch.sparse_coo_tensor(
                self.A_I_indices, self.A_I_values,
                (self.n_neurons, self.n_neurons)
            ).coalesce()
            inh_drive = A_I @ rho

            numerator = logit_rho + self.threshold - w_in_u - exc_drive
            omega = numerator / (inh_drive + 1e-8)
            omega = omega.clamp(min=0.01, max=10.0)

            postsynaptic_ids = self.A_I_indices[1]
            self.A_I_values.mul_(omega[postsynaptic_ids])

            print(f"  One-step design: Ω range [{omega.min():.3f}, {omega.max():.3f}], "
                  f"mean={omega.mean():.3f}")

    def adapt_inhibitory(self, r_t: torch.Tensor, delta: float = None):
        """Inhibitory plasticity (Eq. 7): A^I_ij += δ·(r_i − ρ_i)"""
        if delta is None:
            delta = self.plasticity_delta
        with torch.no_grad():
            r_mean = r_t.mean(dim=0)
            error = r_mean - self.rho
            postsynaptic_ids = self.A_I_indices[1]
            self.A_I_values.add_(delta * error[postsynaptic_ids])

    def adapt_synchronization(self, r_t: torch.Tensor, reward: float, delta: float = 1e-3):
        """
        Reward-gated synchronization learning.

        When the reservoir synchronizes to a mode and gets reward:
        - Strengthen the inhibitory weights that produced that synchronization
        - This makes the same synchronization easier to reach next time

        When no reward:
        - No plasticity (only learn from successful synchronizations)
        """
        if reward <= 0:
            return

        with torch.no_grad():
            r = r_t.squeeze(0)
            # Compute which cluster is winning
            cluster_means = []
            for a in range(self.n_actions):
                weights = self.cluster_membership[:, a]
                cluster_means.append((r * weights).sum() / (weights.sum() + 1e-8))
            cluster_means = torch.tensor(cluster_means, device=self.device)

            # Reward-gated update: strengthen synchronization to winning cluster
            # Increase inhibition on non-winning clusters, decrease on winning cluster
            for a in range(self.n_actions):
                target_neurons = self.cluster_membership[:, a].bool()
                if a == cluster_means.argmax().item():
                    # Winning cluster: reduce inhibition (make it fire more)
                    mask = target_neurons[self.A_I_indices[1]]
                    self.A_I_values[mask] *= (1.0 - delta * reward)
                else:
                    # Losing clusters: increase inhibition (suppress them)
                    mask = target_neurons[self.A_I_indices[1]]
                    self.A_I_values[mask] *= (1.0 + delta * reward * 0.5)

    def self_organize(self, inputs, n_steps=2000, delta_schedule=None):
        """
        Self-organize the reservoir via inhibitory plasticity (Eq. 7).

        Args:
            inputs: sample input vectors for adaptation
            n_steps: total adaptation steps
            delta_schedule: list of (steps, delta) tuples for multi-phase adaptation
        """
        if delta_schedule is None:
            # Default: aggressive then fine-tune
            delta_schedule = [(1000, 0.01), (1000, 1e-3)]

        print(f"\n[Self-Organization] Inhibitory plasticity, {n_steps} steps")
        self.reset_state(batch_size=1)

        with torch.no_grad():
            step = 0
            for phase_steps, delta in delta_schedule:
                print(f"  Phase: δ={delta}, {phase_steps} steps")
                for _ in range(phase_steps):
                    idx = step % len(inputs)
                    u = inputs[idx].unsqueeze(0)
                    r = self.forward(u)
                    self.adapt_inhibitory(r, delta=delta)
                    step += 1

                stats = self.verify_balance()
                print(f"    After {step} steps: |r-ρ|={stats['rho_err']:.4f}  "
                      f"r_mean={stats['r_mean']:.3f}  silent={stats['silent']}  "
                      f"saturated={stats['saturated']}")

    # ── Emergent cluster discovery ────────────────────────────────────

    def discover_clusters(self, n_samples: int = 1000):
        """
        Discover emergent clusters via correlation analysis.

        After self-organization, neurons that fire together for the same inputs
        are in the same functional cluster. We:
        1. Feed random inputs and record firing rates
        2. Compute pairwise correlation between all neurons
        3. Apply k-means on the correlation matrix to find n_actions clusters
        4. Balance cluster sizes
        5. Store soft cluster membership
        """
        from sklearn.cluster import KMeans

        print(f"\n[Cluster Discovery] Analyzing correlations from {n_samples} samples")
        self.reset_state(1)
        rates = []

        with torch.no_grad():
            for _ in range(n_samples):
                u = torch.randn(1, self.input_dim, device=self.device) * 0.5
                r = self.forward(u)
                rates.append(r.squeeze(0).cpu())

        rates = torch.stack(rates)  # [n_samples, n_neurons]

        # Compute correlation matrix
        rates_centered = rates - rates.mean(dim=0, keepdim=True)
        std = rates.std(dim=0) + 1e-8
        correlation = (rates_centered.T @ rates_centered) / n_samples
        correlation = correlation / (std.unsqueeze(0) * std.unsqueeze(1))
        correlation = correlation.numpy()

        # K-means clustering on correlation profiles
        kmeans = KMeans(n_clusters=self.n_actions, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(correlation)
        centers = kmeans.cluster_centers_

        # Balance cluster sizes
        target_size = self.n_neurons // self.n_actions
        cluster_labels = self._balance_clusters(cluster_labels, centers, correlation, target_size)

        # Build cluster membership
        membership = torch.zeros(self.n_neurons, self.n_actions, device=self.device)
        for neuron_id, cluster_id in enumerate(cluster_labels):
            membership[neuron_id, cluster_id] = 1.0

        self.cluster_membership = membership

        sizes = np.bincount(cluster_labels, minlength=self.n_actions)
        print(f"  Discovered {self.n_actions} emergent clusters: sizes={sizes.tolist()}")

        return cluster_labels

    def _balance_clusters(self, labels, centers, correlation, target_size):
        """Rebalance cluster sizes by reassigning neurons."""
        labels = labels.copy()
        n = self.n_neurons
        k = self.n_actions

        for _ in range(100):
            sizes = np.bincount(labels, minlength=k)
            big = int(np.argmax(sizes))
            small = int(np.argmin(sizes))
            if sizes[big] <= target_size + 10:
                break
            if sizes[small] >= target_size - 10:
                break

            big_neurons = np.where(labels == big)[0]
            dists = np.linalg.norm(correlation[big_neurons] - centers[small], axis=1)
            best_idx = big_neurons[np.argmin(dists)]
            labels[best_idx] = small

        return labels

    # ── Generalized synchronization readout ──────────────────────────

    def get_action(self, r_t: torch.Tensor = None) -> tuple:
        """
        Action selection via generalized synchronization readout.

        The reservoir synchronizes to different dynamical modes based on input.
        Each emergent cluster represents one synchronized mode.
        Action = which mode the reservoir is currently synchronized to
                (highest cluster mean = strongest synchronization)
        """
        if r_t is None:
            r_t = self.r

        r_flat = r_t.squeeze(0) if r_t.dim() == 2 else r_t

        # Compute cluster means (weighted by emergent membership)
        cluster_means = []
        for a in range(self.n_actions):
            weights = self.cluster_membership[:, a]
            weighted_sum = (r_flat * weights).sum()
            weight_sum = weights.sum() + 1e-8
            cluster_means.append((weighted_sum / weight_sum).item())

        # Action = argmax of cluster means (which mode is dominant)
        action_id = int(np.argmax(cluster_means))

        return action_id, cluster_means

    # ── Diagnostics ───────────────────────────────────────────────────

    def verify_balance(self) -> dict:
        """Check reservoir health / proximity to criticality."""
        with torch.no_grad():
            r_mean = self.r.squeeze(0)
            rho_err = (r_mean - self.rho).abs().mean().item()
            r_overall = r_mean.mean().item()
            silent = (r_mean < 0.05).sum().item()
            saturated = (r_mean > 0.95).sum().item()

            cluster_means = []
            for a in range(self.n_actions):
                weights = self.cluster_membership[:, a]
                weighted_sum = (r_mean * weights).sum()
                weight_sum = weights.sum() + 1e-8
                cluster_means.append((weighted_sum / weight_sum).item())

        return {
            "rho_err": rho_err,
            "r_mean": r_overall,
            "silent": silent,
            "saturated": saturated,
            "cluster_means": cluster_means,
        }

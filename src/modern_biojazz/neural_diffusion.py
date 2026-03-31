import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple
from modern_biojazz.site_graph import ReactionNetwork, Protein, Rule

class SimpleContactMapDenoiser(nn.Module):
    """
    A very simple MLPMixer-like denoiser for N x N contact maps.
    Predicts the noise added to an adjacency matrix.
    """
    def __init__(self, n_nodes: int, hidden_dim: int = 64, time_embed_dim: int = 32):
        super().__init__()
        self.n_nodes = n_nodes
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        input_dim = n_nodes * n_nodes + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_nodes * n_nodes)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [batch, N, N], t: [batch, 1]
        batch_size = x.size(0)
        t_embed = self.time_mlp(t)  # [batch, time_embed_dim]

        x_flat = x.view(batch_size, -1)  # [batch, N*N]
        h = torch.cat([x_flat, t_embed], dim=-1)  # [batch, N*N + time_embed_dim]

        out_flat = self.net(h)
        return out_flat.view(batch_size, self.n_nodes, self.n_nodes)


class DDPMContactMapTrainer:
    """Trains and samples from a simple continuous DDPM for network contact maps."""
    def __init__(self, n_nodes: int, n_steps: int = 100, device: str = "cpu"):
        self.n_nodes = n_nodes
        self.n_steps = n_steps
        self.device = device

        self.model = SimpleContactMapDenoiser(n_nodes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Beta schedule
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def extract_contact_map(self, network: ReactionNetwork, max_nodes: int) -> torch.Tensor:
        """Converts a network into an N x N adjacency matrix."""
        mat = torch.zeros((max_nodes, max_nodes))
        proteins = sorted(list(network.proteins.keys()))
        if not proteins:
            return mat

        # Map protein names to indices 0..N-1
        p2idx = {p: i for i, p in enumerate(proteins[:max_nodes])}

        for rule in network.rules:
            # For simplicity, extract edges from reactants
            # e.g., kinase -> substrate, a -> b in binding
            if len(rule.reactants) >= 2:
                r1, r2 = rule.reactants[0], rule.reactants[1]
                # Strip suffixes
                r1_base = r1.split('_')[0] if '_' in r1 else r1
                r2_base = r2.split('_')[0] if '_' in r2 else r2

                if r1_base in p2idx and r2_base in p2idx:
                    idx1, idx2 = p2idx[r1_base], p2idx[r2_base]
                    mat[idx1, idx2] = 1.0
                    mat[idx2, idx1] = 1.0  # symmetric for simplicity

        return mat

    def train_step(self, x0: torch.Tensor) -> float:
        """Performs one DDPM training step."""
        self.model.train()
        batch_size = x0.size(0)

        # Sample random t
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        t_float = t.float().unsqueeze(-1) / self.n_steps

        # Add noise
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # Predict noise
        pred_noise = self.model(xt, t_float)

        loss = F.mse_loss(pred_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        """Samples a new contact map from pure noise."""
        self.model.eval()
        x = torch.randn((1, self.n_nodes, self.n_nodes), device=self.device)

        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i], device=self.device)
            t_float = t.float().unsqueeze(-1) / self.n_steps

            z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            pred_noise = self.model(x, t_float)

            alpha_t = self.alpha[t].view(-1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)

            # DDPM reverse step
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
            x = x + torch.sqrt(self.beta[t].view(-1, 1, 1)) * z

        # Threshold to create binary adjacency matrix
        x_bin = (torch.sigmoid(x) > 0.5).float()
        return x_bin.squeeze(0)

    def to_network(self, contact_map: torch.Tensor) -> ReactionNetwork:
        """Converts a binary adjacency matrix back into a basic ReactionNetwork."""
        network = ReactionNetwork()
        n = self.n_nodes

        # Create proteins
        for i in range(n):
            network.proteins[f"P{i}"] = Protein(name=f"P{i}", sites=[])

        # Create binding rules based on edges
        rule_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if contact_map[i, j] > 0.5:
                    p1, p2 = f"P{i}", f"P{j}"
                    comp = f"{p1}:{p2}"
                    network.proteins[comp] = Protein(name=comp, sites=[])
                    network.rules.append(
                        Rule(
                            name=f"bind_{p1}_{p2}_{rule_idx}",
                            rule_type="binding",
                            reactants=[p1, p2],
                            products=[comp],
                            rate=0.1
                        )
                    )
                    rule_idx += 1

        return network

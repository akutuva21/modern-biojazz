import torch
from modern_biojazz.neural_diffusion import SimpleContactMapDenoiser, DDPMContactMapTrainer
from modern_biojazz.site_graph import ReactionNetwork, Protein, Rule

def test_simple_contact_map_denoiser():
    n_nodes = 5
    model = SimpleContactMapDenoiser(n_nodes)

    # [batch_size, n_nodes, n_nodes]
    x = torch.randn((2, n_nodes, n_nodes))
    # [batch_size, 1]
    t = torch.tensor([[0.5], [0.8]])

    out = model(x, t)

    assert out.shape == (2, n_nodes, n_nodes)
    assert not torch.isnan(out).any()

def test_extract_contact_map():
    n_nodes = 3
    trainer = DDPMContactMapTrainer(n_nodes)

    # Create a small network
    net = ReactionNetwork()
    for p in ["A", "B", "C"]:
        net.proteins[p] = Protein(name=p, sites=[])

    net.rules.append(Rule("bind_A_B", "binding", ["A", "B"], ["A:B"], 0.1))

    mat = trainer.extract_contact_map(net, max_nodes=n_nodes)

    assert mat.shape == (3, 3)
    # A is index 0, B is index 1, C is index 2 (alphabetical sorting)
    assert mat[0, 1] == 1.0
    assert mat[1, 0] == 1.0
    assert mat[0, 2] == 0.0
    assert mat[1, 2] == 0.0

def test_extract_contact_map_ignores_extra_nodes():
    n_nodes = 2
    trainer = DDPMContactMapTrainer(n_nodes)
    net = ReactionNetwork()
    for p in ["A", "B", "C"]:
        net.proteins[p] = Protein(name=p, sites=[])

    mat = trainer.extract_contact_map(net, max_nodes=n_nodes)
    assert mat.shape == (2, 2)

def test_to_network():
    n_nodes = 3
    trainer = DDPMContactMapTrainer(n_nodes)

    # Create an adjacency matrix
    mat = torch.zeros((n_nodes, n_nodes))
    mat[0, 1] = 1.0
    mat[1, 0] = 1.0
    mat[1, 2] = 1.0
    mat[2, 1] = 1.0

    net = trainer.to_network(mat)

    assert len(net.proteins) == 5 # 3 base proteins + 2 complexes
    assert len(net.rules) == 2
    assert net.rules[0].reactants == ["P0", "P1"]
    assert net.rules[1].reactants == ["P1", "P2"]

def test_ddpm_train_step():
    n_nodes = 4
    trainer = DDPMContactMapTrainer(n_nodes, n_steps=10)

    x0 = torch.zeros((2, n_nodes, n_nodes))
    x0[0, 0, 1] = 1.0
    x0[0, 1, 0] = 1.0

    loss = trainer.train_step(x0)
    assert isinstance(loss, float)
    assert loss > 0

@torch.no_grad()
def test_ddpm_sample():
    n_nodes = 3
    trainer = DDPMContactMapTrainer(n_nodes, n_steps=5)

    sampled = trainer.sample()
    assert sampled.shape == (n_nodes, n_nodes)
    # The output should be thresholded to 0 or 1
    assert torch.all((sampled == 0.0) | (sampled == 1.0))

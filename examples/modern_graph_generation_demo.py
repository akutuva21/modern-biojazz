import random
import torch
import json
import os
from modern_biojazz.site_graph import ReactionNetwork, Rule, Protein
from modern_biojazz.mutation import GraphMutator
from modern_biojazz.neural_diffusion import DDPMContactMapTrainer
from modern_biojazz.llm_proposer import OpenAICompatibleProposer, LLMDenoisingProposer

def demo_motifs_and_crossover():
    print("=== Demo 1: Motif-Based Actions and Crossover ===")
    mutator = GraphMutator(random.Random(42))

    # 1. Create a baseline network
    net1 = ReactionNetwork()
    mutator.add_protein(net1, "A")
    mutator.add_protein(net1, "B")
    mutator.add_binding_rule(net1, "A", "B")

    print(f"Network 1 initially has {len(net1.proteins)} proteins and {len(net1.rules)} rules.")

    # 2. Add motifs
    mutator.add_kinase_cascade(net1)
    print(f"Added Kinase Cascade. Net 1 now has {len(net1.proteins)} proteins and {len(net1.rules)} rules.")

    # 3. Create a second network
    net2 = ReactionNetwork()
    mutator.add_protein(net2, "X")
    mutator.add_protein(net2, "Y")
    mutator.add_negative_feedback_loop(net2)
    print(f"Network 2 has {len(net2.proteins)} proteins and {len(net2.rules)} rules (Feedback Loop).")

    # 4. Crossover
    child = mutator.crossover(net1, net2)
    print(f"Crossover Child has {len(child.proteins)} proteins and {len(child.rules)} rules.")
    print(f"Child proteins: {list(child.proteins.keys())}")
    print("=" * 50)


def demo_llm_denoising():
    print("\n=== Demo 2: LLM Denoising Diffusion ===")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping LLM Denoising demo. OPENAI_API_KEY environment variable not set.")
        return

    proposer = LLMDenoisingProposer(
        OpenAICompatibleProposer(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model="gpt-4o-mini",
        )
    )

    # Create a noisy/broken network
    noisy_net = ReactionNetwork()
    mutator = GraphMutator()
    mutator.add_protein(noisy_net, "K1")
    mutator.add_protein(noisy_net, "S1")
    # Broken/meaningless interaction
    noisy_net.rules.append(
        Rule(name="broken_bind", rule_type="binding", reactants=["K1", "S1"], products=["K1", "S1"], rate=0.1)
    )

    model_code = (
        f"n_proteins={len(noisy_net.proteins)};"
        f"rules=[r.name for r in noisy_net.rules];"
        f"proteins={list(noisy_net.proteins.keys())}"
    )

    action_names = ["add_site", "add_binding", "remove_rule", "add_phosphorylation"]

    print(f"Sending noisy network to LLM Denoiser. Allowed actions: {action_names}")
    try:
        proposed_actions = proposer.propose(model_code, action_names, budget=2)
        print(f"LLM Denoiser proposed repair actions: {proposed_actions}")
    except Exception as e:
        print(f"Failed to query LLM: {e}")
    print("=" * 50)


def demo_pytorch_diffusion():
    print("\n=== Demo 3: PyTorch Neural Diffusion (Contact Maps) ===")

    n_nodes = 5
    n_steps = 100
    trainer = DDPMContactMapTrainer(n_nodes=n_nodes, n_steps=n_steps)
    mutator = GraphMutator(random.Random(42))

    # 1. Synthesize a training dataset of 50 small reaction networks
    print(f"Synthesizing 50 random {n_nodes}-node networks...")
    dataset = []
    for _ in range(50):
        net = ReactionNetwork()
        for i in range(n_nodes):
            mutator.add_protein(net, f"P{i}")

        # Add random bindings
        for _ in range(3):
            p1, p2 = random.sample([f"P{i}" for i in range(n_nodes)], 2)
            mutator.add_binding_rule(net, p1, p2)

        mat = trainer.extract_contact_map(net, max_nodes=n_nodes)
        dataset.append(mat)

    dataset_tensor = torch.stack(dataset)

    # 2. Train the DDPM
    print("Training DDPM for 200 epochs...")
    epochs = 200
    for epoch in range(epochs):
        loss = trainer.train_step(dataset_tensor)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

    # 3. Sample a novel contact map
    print("Sampling new contact map from noise...")
    sampled_map = trainer.sample()
    print("Sampled Adjacency Matrix:")
    print(sampled_map.numpy())

    # 4. Map back to BNGL/ReactionNetwork
    generated_net = trainer.to_network(sampled_map)
    print(f"\nGenerated ReactionNetwork has {len(generated_net.proteins)} proteins and {len(generated_net.rules)} binding rules.")
    print("Rules extracted from diffusion model:")
    for r in generated_net.rules:
        print(f"  {r.name}: {r.reactants} -> {r.products}")
    print("=" * 50)

if __name__ == "__main__":
    demo_motifs_and_crossover()
    demo_llm_denoising()
    demo_pytorch_diffusion()

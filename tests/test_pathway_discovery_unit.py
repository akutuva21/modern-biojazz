import json
from pathlib import Path
from modern_biojazz.pathway_discovery import (
    PathwayDiscoveryResult,
    load_discovery_snapshot,
    save_discovery_snapshot,
)

def test_save_and_load_discovery_snapshot(tmp_path: Path):
    result = PathwayDiscoveryResult(
        seed_genes=["A", "B"],
        species=["A", "B", "C"],
        interactions=[{"source_genesymbol": "A", "target_genesymbol": "B"}],
        source="omnipath"
    )

    file_path = tmp_path / "snapshot.json"
    save_discovery_snapshot(result, str(file_path))

    loaded_result = load_discovery_snapshot(str(file_path))

    assert loaded_result.seed_genes == result.seed_genes
    assert loaded_result.species == result.species
    assert loaded_result.interactions == result.interactions
    assert loaded_result.source == result.source

def test_load_discovery_snapshot_missing_source(tmp_path: Path):
    data = {
        "seed_genes": ["X", "Y"],
        "species": ["X", "Y", "Z"],
        "interactions": []
    }

    file_path = tmp_path / "missing_source_snapshot.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_result = load_discovery_snapshot(str(file_path))

    assert loaded_result.seed_genes == data["seed_genes"]
    assert loaded_result.species == data["species"]
    assert loaded_result.interactions == data["interactions"]
    assert loaded_result.source == "snapshot"

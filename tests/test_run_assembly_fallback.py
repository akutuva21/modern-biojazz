from unittest.mock import patch
from modern_biojazz.e2e_pipeline import _run_assembly, E2EConfig
from modern_biojazz.pathway_discovery import PathwayDiscoveryResult
from modern_biojazz.indra_assembly import AssemblyResult

def test_run_assembly_exception_fallback():
    """Test that _run_assembly falls back to minimal BNGL when INDRAAssembler raises an exception."""
    cfg = E2EConfig()
    discovery = PathwayDiscoveryResult(
        seed_genes=["A", "B"],
        species=["A", "B"],
        interactions=[],
        source="test"
    )

    with patch("modern_biojazz.e2e_pipeline.INDRAAssembler") as mock_assembler_cls:
        mock_assembler = mock_assembler_cls.return_value
        mock_assembler.assemble.side_effect = Exception("Test Error")

        result = _run_assembly(cfg, discovery)

        # Verify fallback result
        assert result.source == "offline_fallback::Test Error"
        assert result.species == ["A", "B"]
        assert "begin molecule types" in result.bngl_text
        assert "A()" in result.bngl_text
        assert "B()" in result.bngl_text

def test_run_assembly_no_statements_fallback():
    """Test that _run_assembly falls back to minimal BNGL when INDRA returns no statements."""
    cfg = E2EConfig()
    discovery = PathwayDiscoveryResult(
        seed_genes=["A", "B"],
        species=["A", "B"],
        interactions=[],
        source="test"
    )

    with patch("modern_biojazz.e2e_pipeline.INDRAAssembler") as mock_assembler_cls:
        mock_assembler = mock_assembler_cls.return_value
        # Mocking assemble to return an AssemblyResult with no statements
        mock_assembler.assemble.return_value = AssemblyResult(
            species=["A", "B"],
            statements=[],
            bngl_text="some text that should be ignored",
            source="indra_live"
        )

        result = _run_assembly(cfg, discovery)

        # Verify fallback result
        assert result.source == "offline_fallback::INDRA returned no statements"
        assert result.species == ["A", "B"]
        assert "begin molecule types" in result.bngl_text
        assert "A()" in result.bngl_text
        assert "B()" in result.bngl_text

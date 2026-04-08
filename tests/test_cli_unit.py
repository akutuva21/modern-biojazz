import json
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from modern_biojazz.cli import main


@pytest.fixture
def mock_open_json():
    with patch("builtins.open", mock_open(read_data='{}')) as m_open:
        with patch("json.load", return_value={}) as m_load:
            yield m_open, m_load


@pytest.fixture
def mock_dependencies(mock_open_json):
    with patch("modern_biojazz.cli.ReactionNetwork.from_dict") as m_rn:
        with patch("modern_biojazz.cli.ModernBioJazzPipeline") as m_pipeline_cls:
            with patch("builtins.print") as m_print:
                mock_pipeline_instance = MagicMock()
                m_pipeline_cls.return_value = mock_pipeline_instance
                mock_result = MagicMock()
                mock_result.evolution.best_score = 1.0
                mock_result.evolution.history = []
                mock_result.evolution.best_network.to_dict.return_value = {}
                mock_result.grounding = None
                mock_pipeline_instance.run.return_value = mock_result
                yield mock_pipeline_instance


def test_main_missing_seed():
    with patch("sys.argv", ["cli"]):
        with pytest.raises(SystemExit):
            main()


def test_main_default(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json"]):
        main()
        mock_dependencies.run.assert_called_once()


def test_main_sim_http_missing_url(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--sim-backend", "http"]):
        with pytest.raises(ValueError, match="--sim-base-url is required when --sim-backend=http"):
            main()


def test_main_sim_http_with_url(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--sim-backend", "http", "--sim-base-url", "http://localhost"]):
        main()


def test_main_llm_openai_missing_url(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--llm-provider", "openai_compatible"]):
        with pytest.raises(ValueError, match="--llm-base-url is required when --llm-provider=openai_compatible"):
            main()


def test_main_llm_openai_missing_api_key(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--llm-provider", "openai_compatible", "--llm-base-url", "http://localhost"]):
        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="Environment variable OPENAI_API_KEY must be set for llm provider"):
                main()


def test_main_llm_openai_with_key(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--llm-provider", "openai_compatible", "--llm-base-url", "http://localhost"]):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            main()


def test_main_llm_deterministic(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--llm-provider", "deterministic"]):
        main()


def test_main_with_grounding(mock_dependencies):
    with patch("sys.argv", ["cli", "--seed", "seed.json", "--grounding", "grounding.json"]):
        mock_result = MagicMock()
        mock_result.evolution.best_score = 1.0
        mock_result.evolution.history = []
        mock_result.evolution.best_network.to_dict.return_value = {}

        mock_grounding = MagicMock()
        mock_grounding.mapping = {}
        mock_grounding.score = 1.0
        mock_grounding.candidates_considered = 1
        mock_result.grounding = mock_grounding

        mock_dependencies.run.return_value = mock_result
        main()

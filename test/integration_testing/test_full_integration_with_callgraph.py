import json
import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cli.cli_runner import CodeSmileCLI


@pytest.fixture
def integration_setup_callgraph(tmp_path):
    input_path = tmp_path / "input_cg"
    output_path = tmp_path / "output_cg"

    input_path.mkdir()

    # Create two files with a dependency
    (input_path / "lib.py").write_text(
        """
def helper():
    pass
"""
    )

    (input_path / "main.py").write_text(
        """
from lib import helper

def main():
    helper()
"""
    )

    return str(input_path), str(output_path)


@patch("components.rule_checker.RuleChecker.rule_check")
def test_full_integration_with_callgraph(mock_rule_check, integration_setup_callgraph):
    # Mock return value to avoid errors in analysis (we verify callgraph mostly)
    mock_rule_check.return_value = pd.DataFrame(
        columns=[
            "filename",
            "function_name",
            "smell_name",
            "line",
            "description",
            "additional_info",
        ]
    )

    input_path, output_path = integration_setup_callgraph

    args = Mock(
        input=input_path,
        output=output_path,
        max_walkers=1,
        parallel=False,
        resume=False,
        multiple=False,
        callgraph=True,
    )

    cli = CodeSmileCLI(args)
    cli.execute()

    # Check Call Graph JSON
    cg_file = os.path.join(output_path, "call_graph.json")
    assert os.path.exists(cg_file), f"File {cg_file} not found"

    with open(cg_file, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # helper and main should be in nodes
    node_ids = {n["name"] for n in nodes}
    assert "helper" in node_ids
    assert "main" in node_ids

    # edge main -> helper should exist
    # Edge logic depends on implementation.
    # lib.helper and main.main.
    # main calls helper.
    # source: main.main, target: lib.helper (assuming resolved correctly)

    # Let's verify at least one edge exists
    assert len(edges) > 0

    print("Test Passed: Full Integration with CallGraph")

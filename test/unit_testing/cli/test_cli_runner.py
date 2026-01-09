from unittest.mock import MagicMock, patch
import runpy
import os

import pytest

from cli.cli_runner import CodeSmileCLI


# Mock the ProjectAnalyzer class for testing
@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock()
    return analyzer


# Test that the execute method runs without errors for valid arguments
def test_execute_with_valid_arguments(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = False
    args.max_walkers = 5

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_project.return_value = 2  # Assume it finds 2 code smells
    mock_analyzer.clean_output_directory = MagicMock()

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        # Ensure the methods were called as expected
        mock_analyzer.analyze_project.assert_called_once_with("mock_input")
        mock_print.assert_any_call("Analysis completed. Total code smells found: 2")


# Test that the execute method raises an error
# for missing input or output arguments
def test_execute_with_missing_arguments():
    args = MagicMock()
    args.input = None  # Missing input argument
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = False
    args.max_walkers = 5

    # Initialize the CLI with mocked arguments
    cli = CodeSmileCLI(args)

    with patch("builtins.print") as mock_print:
        with pytest.raises(
            SystemExit
        ):  # This will raise a SystemExit due to invalid arguments
            cli.execute()

        mock_print.assert_any_call(
            "Error: Please specify both input and output folders."
        )


# Test for handling invalid max_walkers argument with parallel execution
def test_execute_with_invalid_max_walkers(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = True
    args.max_walkers = -1  # Invalid max_walkers
    args.resume = False
    args.multiple = True

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_projects_parallel = MagicMock()

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with pytest.raises(ValueError, match="max_walkers must be greater than 0."):
        cli.execute()


# Test for parallel execution with multiple projects
def test_execute_with_parallel_execution(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = True
    args.resume = False
    args.multiple = True
    args.max_walkers = 5

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_projects_parallel.return_value = (
        None  # Assume no return value
    )
    mock_analyzer.clean_output_directory = MagicMock()
    mock_analyzer.merge_all_results = MagicMock()

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        # Ensure parallel execution method was called
        mock_analyzer.analyze_projects_parallel.assert_called_once_with("mock_input", 5)
        mock_analyzer.merge_all_results.assert_called_once()
        mock_print.assert_any_call("Analysis results saved successfully.")


# Test for sequential execution with one project
def test_execute_with_sequential_execution(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = False
    args.max_walkers = 5

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_project.return_value = 2  # Assume it finds 2 code smells

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        # Ensure sequential execution method was called
        mock_analyzer.analyze_project.assert_called_once_with("mock_input")
        mock_print.assert_any_call("Analysis completed. Total code smells found: 2")


# Test for handling resume functionality
def test_execute_with_resume(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = True
    args.multiple = False
    args.max_walkers = 5

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_project.return_value = 2
    mock_analyzer.clean_output_directory = MagicMock()

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        # Check that clean_output_directory was not called due to resume
        mock_analyzer.clean_output_directory.assert_not_called()
        mock_analyzer.analyze_project.assert_called_once_with("mock_input")
        mock_print.assert_any_call("Analysis completed. Total code smells found: 2")


def test_print_configuration(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = False
    args.max_walkers = 5

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_project.return_value = 2

    # Initialize the CLI with mocked arguments
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        # Test the print statements for configuration
        mock_print.assert_any_call(
            "Starting analysis with the following configuration:"
        )
        mock_print.assert_any_call(f"Input folder: {args.input}")
        mock_print.assert_any_call(f"Output folder: {args.output}")
        mock_print.assert_any_call(f"Parallel execution: {args.parallel}")
        mock_print.assert_any_call(f"Resume execution: {args.resume}")
        mock_print.assert_any_call(f"Max Walkers: {args.max_walkers}")
        mock_print.assert_any_call(f"Analyze multiple projects: {args.multiple}")


def test_execute_with_resume_and_multiple_projects(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = True  # Parallel execution enabled
    args.resume = True  # Resume execution
    args.multiple = True  # Multiple projects flag set
    args.max_walkers = 5

    # Mock the methods of ProjectAnalyzer
    mock_analyzer.analyze_projects_parallel.return_value = (
        None  # Assume no return value
    )
    mock_analyzer.clean_output_directory = MagicMock()
    mock_analyzer.merge_all_results = MagicMock()

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        # Ensure clean_output_directory is not called due to resume
        mock_analyzer.clean_output_directory.assert_not_called()
        # Ensure merge_all_results is
        # called because multiple projects are being analyzed
        mock_analyzer.merge_all_results.assert_called_once()
        mock_print.assert_any_call("Analysis results saved successfully.")


def test_execute_with_invalid_max_walkers_and_parallel(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = True  # Parallel execution enabled
    args.max_walkers = 0  # Invalid max_walkers
    args.resume = False
    args.multiple = False

    # Initialize the CLI with mocked arguments and analyzer
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer  # Inject the mock analyzer

    with pytest.raises(ValueError, match="max_walkers must be greater than 0."):
        cli.execute()


def test_execute_with_callgraph(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = False
    args.max_walkers = 5
    args.callgraph = True

    mock_analyzer.analyze_project.return_value = 2
    mock_analyzer.generate_call_graph = MagicMock()

    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        mock_analyzer.analyze_project.assert_called_once_with("mock_input")
        mock_analyzer.generate_call_graph.assert_called_once_with("mock_input")


def test_execute_with_callgraph_and_multiple(mock_analyzer):
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = True
    args.max_walkers = 5
    args.callgraph = True

    mock_analyzer.analyze_projects_parallel.return_value = None
    mock_analyzer.merge_all_results.return_value = None
    mock_analyzer.generate_call_graph = MagicMock()

    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer

    with patch("builtins.print") as mock_print:
        cli.execute()

        mock_analyzer.generate_call_graph.assert_not_called()
        mock_print.assert_any_call(
            "Call Graph generation is not supported for multiple projects mode yet."
        )


def test_execute_with_multiple_projects_sequential(mock_analyzer):
    """
    Test for sequential execution with multiple projects (parallel=False).
    """
    args = MagicMock()
    args.input = "mock_input"
    args.output = "mock_output"
    args.parallel = False
    args.resume = False
    args.multiple = True
    args.max_walkers = 5
    args.callgraph = False
    
    # Mock analyzer methods
    mock_analyzer.analyze_projects_sequential = MagicMock()
    mock_analyzer.clean_output_directory = MagicMock()
    mock_analyzer.merge_all_results = MagicMock()
    
    cli = CodeSmileCLI(args)
    cli.analyzer = mock_analyzer
    
    with patch("builtins.print") as mock_print:
        cli.execute()
        
        mock_analyzer.analyze_projects_sequential.assert_called_once_with("mock_input", resume=False)
        mock_analyzer.merge_all_results.assert_called_once()
        mock_print.assert_any_call("Analysis results saved successfully.")


def test_main_success():
    """
    Test clean execution of main() with valid arguments.
    """
    with patch("sys.argv", ["cli_runner.py", "--input", "in", "--output", "out"]):
        with patch("cli.cli_runner.CodeSmileCLI") as MockCLI:
            mock_instance = MockCLI.return_value
            mock_instance.execute = MagicMock()
            
            from cli.cli_runner import main
            main()
            
            MockCLI.assert_called_once()
            mock_instance.execute.assert_called_once()


def test_main_missing_arguments():
    """
    Test main() raises SystemExit when args are missing.
    """
    # Simulate missing arguments (only script name)
    with patch("sys.argv", ["cli_runner.py"]):
        from cli.cli_runner import main
        
        # Expect SystemExit(1)
        with pytest.raises(SystemExit) as excinfo:
            # Also catch print calls to verify error message if possible, 
            # though argparse prints to stderr usually.
             with patch("builtins.print") as mock_print:
                 main()
        
        assert excinfo.value.code == 1


def test_script_execution():
    """
    Test that the script runs when executed as __main__.
    Covers the 'if __name__ == "__main__":' block.
    """
    # Patch sys.argv to simulate running with arguments
    with patch("sys.argv", ["cli_runner.py", "--input", "mock_in", "--output", "mock_out"]):
        # We cannot patch CodeSmileCLI because runpy executes the class definition again.
        # Instead, we patch the dependencies it uses, specifically ProjectAnalyzer.
        with patch("components.project_analyzer.ProjectAnalyzer") as MockAnalyzerCls:
            mock_analyzer_instance = MockAnalyzerCls.return_value
            mock_analyzer_instance.analyze_project.return_value = 0
            
            # Use runpy to execute the module as if it were a script
            from cli import cli_runner
            runpy.run_path(cli_runner.__file__, run_name="__main__")
            
            # Verify ProjectAnalyzer was instantiated and used
            MockAnalyzerCls.assert_called()
            mock_analyzer_instance.analyze_project.assert_called()

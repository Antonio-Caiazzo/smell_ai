import os
import shutil
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from components.project_analyzer import ProjectAnalyzer


@pytest.fixture
def mock_output_path(tmp_path):
    """
    Pytest fixture to create a temporary output directory.
    """
    return str(tmp_path)


@pytest.fixture
def project_analyzer(mock_output_path):
    """
    Fixture to create an instance of ProjectAnalyzer.
    """
    return ProjectAnalyzer(output_path=mock_output_path)


@pytest.fixture
def mock_file_related_methods(monkeypatch):
    """
    Fixture to mock the file-related methods.
    This fixture reduces repetition
    for mocking methods like os.path, FileUtils, etc.
    """
    monkeypatch.setattr("os.path.isdir", lambda path: True)
    monkeypatch.setattr("os.listdir", lambda path: ["project1", "project2"])
    monkeypatch.setattr(
        "utils.file_utils.FileUtils.get_python_files",
        lambda path: ["file1.py"],
    )
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda path: None)
    monkeypatch.setattr(
        "utils.file_utils.FileUtils.synchronized_append_to_log",
        lambda path, project, lock: None,
    )


def test_analyze_project(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test the `analyze_project` method.
    """

    output_dir = tmp_path / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "components.project_analyzer.ProjectAnalyzer._save_results",
        lambda self, df, path: df.to_csv(output_dir / "overview.csv", index=False),
    )

    # Mock inspection results for two files
    mock_inspection_results = [
        pd.DataFrame(
            {
                "filename": ["file1.py"],
                "function_name": ["func1"],
                "smell_name": ["smell1"],
                "line": [10],
                "description": ["desc1"],
                "additional_info": ["info1"],
            }
        ),
        pd.DataFrame(
            {
                "filename": ["file2.py"],
                "function_name": ["func2"],
                "smell_name": ["smell2"],
                "line": [20],
                "description": ["desc2"],
                "additional_info": ["info2"],
            }
        ),
    ]

    # Mock inspect method to return the inspection results
    project_analyzer.inspector.inspect = MagicMock(side_effect=mock_inspection_results)

    # Mock the get_python_files method to return both files
    monkeypatch.setattr(
        "utils.file_utils.FileUtils.get_python_files",
        lambda _: ["file1.py", "file2.py"],
    )

    # Run the method
    total_smells = project_analyzer.analyze_project(
        "test/unit_testing/components/mock_project_path"
    )

    # Assertions
    assert total_smells == 2  # Expecting 2 smells (from file1.py and file2.py)
    project_analyzer.inspector.inspect.assert_any_call("file1.py")
    project_analyzer.inspector.inspect.assert_any_call("file2.py")

    mock_project_path = "test/unit_testing/components/mock_project_path"
    if os.path.exists(mock_project_path):
        shutil.rmtree(mock_project_path)


def test_analyze_projects_sequential(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test the `analyze_projects_sequential` method.
    """

    output_dir = tmp_path / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "components.project_analyzer.ProjectAnalyzer._save_results",
        lambda self, df, path: df.to_csv(output_dir / "overview.csv", index=False),
    )

    # Mock the inspector's inspect method
    mock_inspection_results = pd.DataFrame(
        {
            "filename": ["file1.py"],
            "function_name": ["func1"],
            "smell_name": ["smell1"],
            "line": [10],
        }
    )
    project_analyzer.inspector.inspect = MagicMock(return_value=mock_inspection_results)

    # Call the method
    project_analyzer.analyze_projects_sequential(
        "test/unit_testing/components/mock_project_path", resume=False
    )

    # Ensure inspect was called
    project_analyzer.inspector.inspect.assert_called_with("file1.py")

    mock_project_path = "test/unit_testing/components/mock_project_path"
    if os.path.exists(mock_project_path):
        shutil.rmtree(mock_project_path)


def test_clean_output_directory(monkeypatch, project_analyzer):
    """
    Test the `clean_output_directory` method.
    """
    mock_clean_directory = MagicMock()
    monkeypatch.setattr(
        "utils.file_utils.FileUtils.clean_directory", mock_clean_directory
    )

    # Run the method
    project_analyzer.clean_output_directory()

    # Assertions
    mock_clean_directory.assert_called_once_with(project_analyzer.output_path)


def test_merge_all_results(monkeypatch, project_analyzer):
    """
    Test the `merge_all_results` method.
    """
    mock_merge_results = MagicMock()
    monkeypatch.setattr("utils.file_utils.FileUtils.merge_results", mock_merge_results)

    # Run the method
    project_analyzer.merge_all_results()

    # Assertions
    mock_merge_results.assert_called_once_with(
        input_dir=os.path.join(project_analyzer.output_path, "project_details"),
        output_dir=project_analyzer.output_path,
    )


def test_analyze_projects_parallel(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test the `analyze_projects_parallel` method.
    """

    mock_inspection_results = pd.DataFrame(
        {
            "filename": ["file1.py"],
            "function_name": ["func1"],
            "smell_name": ["smell1"],
            "line": [10],
            "description": ["desc1"],
            "additional_info": ["info1"],
        }
    )

    # Mock dependencies
    monkeypatch.setattr(
        "os.path.exists", lambda path: True  # Mock that all paths exist
    )
    monkeypatch.setattr(
        "os.path.isdir",
        lambda path: True,  # Mock that all paths are directories
    )

    # Mock the inspector's inspect method
    project_analyzer.inspector.inspect = MagicMock(return_value=mock_inspection_results)

    # Mock save results method
    monkeypatch.setattr(
        "components.project_analyzer.ProjectAnalyzer._save_results",
        lambda self, df, path: None,  # Do nothing on saving results
    )

    # Mock ThreadPoolExecutor to avoid threading and run tasks synchronously
    with patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor:
        mock_executor = MagicMock()
        MockExecutor.return_value = mock_executor
        mock_executor.__enter__.return_value = mock_executor
        # Make sure the function gets executed immediately (synchronously)
        mock_executor.submit.side_effect = lambda func, *args, **kwargs: func(
            *args, **kwargs
        )

        # Run the method
        with patch("builtins.print") as mock_print:
            project_analyzer.analyze_projects_parallel(
                "test/unit_testing/components/mock_base_path", max_workers=1
            )

        # Ensure the inspector's inspect method
        # was called the expected number of times
        assert project_analyzer.inspector.inspect.call_count == 2

        # Check if print statements were made (optional)
        assert mock_print.call_count > 0


def test_exception_handling_in_inspect(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test that the `inspect` method handles exceptions gracefully.
    """

    # Simulate an exception in the inspect method
    project_analyzer.inspector.inspect = MagicMock(side_effect=FileNotFoundError)

    with patch("builtins.print") as mock_print:
        project_analyzer.analyze_projects_parallel(
            "test/unit_testing/components/mock_project_path", max_workers=1
        )

    # Assertions
    assert "Total code smells found in all projects: 0\n" in mock_print.call_args[0][0]

    mock_project_path = "test/unit_testing/components/mock_project_path"
    if os.path.exists(mock_project_path):
        shutil.rmtree(mock_project_path)


def test_analyze_project_with_errors(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test `analyze_project` with error
    handling (FileNotFoundError, SyntaxError).
    """
    output_dir = tmp_path / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "components.project_analyzer.ProjectAnalyzer._save_results",
        lambda self, df, path: df.to_csv(output_dir / "overview.csv", index=False),
    )

    # Mocking a SyntaxError for a specific file
    project_analyzer.inspector.inspect = MagicMock(side_effect=SyntaxError)

    # Run the method (simulate failure for file1.py)
    project_analyzer.analyze_project("test/unit_testing/components/mock_project_path")

    # Check if the error is logged to the error.txt file
    # ProjectAnalyzer writes error.txt to self.output_path (tmp_path)
    error_file = tmp_path / "error.txt"
    with open(error_file, "r") as f:
        error_content = f.read()

    assert "Error in file file1.py: " in error_content  # Check that error is logged

    mock_project_path = "test/unit_testing/components/mock_project_path"
    if os.path.exists(mock_project_path):
        shutil.rmtree(mock_project_path)


def test_analyze_projects_sequential_save_results(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test saving results in `project_details` for sequential analysis.
    """
    output_dir = tmp_path / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "components.project_analyzer.ProjectAnalyzer._save_results",
        lambda self, df, path: df.to_csv(output_dir / "overview.csv", index=False),
    )

    # Mock the inspector's inspect method
    mock_inspection_results = pd.DataFrame(
        {
            "filename": ["file1.py"],
            "function_name": ["func1"],
            "smell_name": ["smell1"],
            "line": [10],
        }
    )
    project_analyzer.inspector.inspect = MagicMock(return_value=mock_inspection_results)

    # Call the method
    project_analyzer.analyze_projects_sequential(
        "test/unit_testing/components/mock_project_path", resume=False
    )

    # Check if project_details directory and the result file were created
    # ProjectAnalyzer writes project_details to self.output_path (tmp_path)
    details_path = tmp_path / "project_details"
    assert details_path.exists()

    detailed_file_path = details_path / "project1_results.csv"
    assert detailed_file_path.exists()

    # Check if the CSV file contains the expected data
    df = pd.read_csv(detailed_file_path)
    assert not df.empty
    assert "filename" in df.columns
    assert df["filename"].iloc[0] == "file1.py"

    mock_project_path = "test/unit_testing/components/mock_project_path"
    if os.path.exists(mock_project_path):
        shutil.rmtree(mock_project_path)


def test_analyze_projects_parallel_thread_safety(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test thread-safety in the `analyze_projects_parallel` method.
    """

    mock_inspection_results = pd.DataFrame(
        {
            "filename": ["file1.py"],
            "function_name": ["func1"],
            "smell_name": ["smell1"],
            "line": [10],
            "description": ["desc1"],
            "additional_info": ["info1"],
        }
    )

    # Mock the inspector's inspect method
    project_analyzer.inspector.inspect = MagicMock(return_value=mock_inspection_results)

    # Mock the synchronized_append_to_log method to check for thread-safety
    mock_synchronized_append = MagicMock()
    monkeypatch.setattr(
        "utils.file_utils.FileUtils.synchronized_append_to_log",
        mock_synchronized_append,
    )

    # Run the method with parallel execution
    project_analyzer.analyze_projects_parallel(
        "test/unit_testing/components/mock_base_path", max_workers=2
    )

    # Normalize the paths for cross-platform consistency
    expected_path = os.path.join(
        "test/unit_testing/components/mock_base_path", "execution_log.txt"
    )

    # Ensure the synchronized_append_to_log
    # method was called with both project1 and project2
    mock_synchronized_append.assert_any_call(expected_path, "project1", ANY)
    mock_synchronized_append.assert_any_call(expected_path, "project2", ANY)

    mock_project_path = "test/unit_testing/components/mock_base_path"
    if os.path.exists(mock_project_path):
        shutil.rmtree(mock_project_path)


def test_analyze_project_empty_directory(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test `analyze_project` when no Python files exist in the directory.
    Should raise ValueError with appropriate message.
    """
    output_dir = tmp_path / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Create a mock FileUtils class that returns empty list
    class MockFileUtils:
        @staticmethod
        def get_python_files(path):
            return []

    monkeypatch.setattr(
        "components.project_analyzer.ProjectAnalyzer._save_results",
        lambda self, df, path: df.to_csv(output_dir / "overview.csv", index=False),
    )

    # Mock FileUtils.get_python_files at the module level where it's imported
    from components import project_analyzer as pa_module

    original_file_utils = pa_module.FileUtils
    pa_module.FileUtils = MockFileUtils

    try:
        # Run the method - should raise ValueError for empty directory
        with pytest.raises(ValueError, match="contains no Python files"):
            project_analyzer.analyze_project(
                "test/unit_testing/components/mock_project_path"
            )
    finally:
        # Restore original FileUtils
        pa_module.FileUtils = original_file_utils


def test_generate_call_graph(
    monkeypatch, project_analyzer, mock_file_related_methods, tmp_path
):
    """
    Test the `generate_call_graph` method.
    """
    output_dir = tmp_path / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Mock CallGraphGenerator
    mock_generator_instance = MagicMock()
    mock_generator_instance.generate.return_value = {"nodes": [], "edges": []}
    mock_generator_instance.generate_dot.return_value = "digraph G {}"
    mock_CallGraphGenerator = MagicMock(return_value=mock_generator_instance)

    # Patch the class where it's imported in project_analyzer.py
    with patch(
        "components.project_analyzer.CallGraphGenerator", mock_CallGraphGenerator
    ):
        project_analyzer.generate_call_graph("mock_project_path")

        # Verify Generator init and generate calls
        mock_CallGraphGenerator.assert_called_once_with("mock_project_path")
        # FileUtils.get_python_files is mocked by fixture to return ["file1.py"]
        mock_generator_instance.generate.assert_called_once_with(["file1.py"])

        # Verify output file creation
        # ProjectAnalyzer writes to self.output_path which is str(tmp_path)
        expected_file = tmp_path / "call_graph.json"
        assert expected_file.exists()


def test_save_results_empty_returns_early(monkeypatch, project_analyzer, tmp_path):
    """
    Test that _save_results returns early when dataframe is empty.
    """
    with patch("os.makedirs") as mock_makedirs:
        project_analyzer._save_results(pd.DataFrame(), "test.csv")
        mock_makedirs.assert_not_called()


def test_analyze_projects_sequential_creates_base_path(monkeypatch, project_analyzer, tmp_path):
    """
    Test that analyze_projects_sequential creates base_path if not exists.
    """
    target_base = tmp_path / "non_existent_base"
    
    # Mock os.listdir to empty list to avoid processing loop
    monkeypatch.setattr("os.listdir", lambda p: [])
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda p: None)

    project_analyzer.analyze_projects_sequential(str(target_base))
    
    assert target_base.exists()


def test_analyze_projects_sequential_resume_logic(monkeypatch, project_analyzer, mock_file_related_methods):
    """
    Test resume logic: skip projects <= last_logged_project.
    """
    monkeypatch.setattr("utils.file_utils.FileUtils.get_last_logged_project", lambda p: "project1")
    monkeypatch.setattr("os.listdir", lambda p: ["project1", "project2"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    
    # Spy on inspect
    project_analyzer.inspector.inspect = MagicMock(return_value=pd.DataFrame()) 
    
    project_analyzer.analyze_projects_sequential("mock_base", resume=True)
    
    # inspect should be called only for project2's files.
    # The fixture mock_file_related_methods sets get_python_files -> ["file1.py"]
    # So if project1 is skipped, inspect is called once for project2.
    assert project_analyzer.inspector.inspect.call_count == 1


def test_analyze_projects_sequential_skips_nondirs(monkeypatch, project_analyzer, mock_file_related_methods):
    """
    Test that analyze_projects_sequential skips non-directory items.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["file.txt"])
    monkeypatch.setattr("os.path.isdir", lambda p: False)
    
    project_analyzer.inspector.inspect = MagicMock()
    
    project_analyzer.analyze_projects_sequential("mock_base")
    
    project_analyzer.inspector.inspect.assert_not_called()


def test_analyze_projects_sequential_handles_outer_exception(monkeypatch, project_analyzer):
    """
    Test that the loop handles generic exceptions and continues.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["proj1"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    
    # Raise generic exception
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", MagicMock(side_effect=Exception("Boom")))
    
    with patch("builtins.print") as mock_print:
        project_analyzer.analyze_projects_sequential("mock_base")
        # Check that error is logged
        found_error = any("Error analyzing project 'proj1': Boom" in str(c) for c in mock_print.call_args_list)
        assert found_error


def test_analyze_projects_sequential_no_smells_no_detailed_save(monkeypatch, project_analyzer, mock_file_related_methods):
    """
    Test that if no smells are found, no detailed CSV is saved.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["proj1"])
    
    # Inspect returns empty
    project_analyzer.inspector.inspect = MagicMock(return_value=pd.DataFrame(columns=["filename"]))
    
    # Patch DataFrame.to_csv at the class level to capture calls
    with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
         project_analyzer.analyze_projects_sequential("mock_base")
         mock_to_csv.assert_not_called()


def test_analyze_projects_parallel_creates_dirs(monkeypatch, project_analyzer, tmp_path):
    """
    Test that analyze_projects_parallel creates base_path if not exists.
    """
    target_base = tmp_path / "par_base"
    monkeypatch.setattr("os.listdir", lambda p: [])
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda p: None)
    
    with patch("concurrent.futures.ThreadPoolExecutor"):
        project_analyzer.analyze_projects_parallel(str(target_base), 1)
    
    assert target_base.exists()


def test_analyze_projects_parallel_skips_excluded(monkeypatch, project_analyzer):
    """
    Test skipping of output folder, logs, and non-directories in parallel analysis.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["output", "execution_log.txt", "file.txt"])
    
    def isdir_side_effect(path):
        if "file.txt" in path: return False
        return True
    
    monkeypatch.setattr("os.path.isdir", isdir_side_effect)
    
    mock_files = MagicMock()
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", mock_files)
    
    # Run synchronously
    with patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor:
        mock_executor = MagicMock()
        MockExecutor.return_value = mock_executor
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        project_analyzer.analyze_projects_parallel("mock_base", 1)
        
    mock_files.assert_not_called()


def test_analyze_projects_parallel_print_smells_found(monkeypatch, project_analyzer):
    """
    Test that 'Found X code smells' is printed when count > 0.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["proj1"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", lambda p: ["f.py"])
    
    # Mock inspect to return list of 3 items
    mock_res = MagicMock()
    mock_res.__len__.return_value = 3
    project_analyzer.inspector.inspect = MagicMock(return_value=mock_res)
    
    # Handle pd.concat
    monkeypatch.setattr("pandas.concat", lambda objs, ignore_index: pd.DataFrame())
    
    # Mock save results
    monkeypatch.setattr("components.project_analyzer.ProjectAnalyzer._save_results", lambda self, df, path: None)
    # Mock thread-safe log
    monkeypatch.setattr("utils.file_utils.FileUtils.synchronized_append_to_log", lambda path, proj, lock: None)
    
    with patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor:
        mock_executor = MagicMock()
        MockExecutor.return_value = mock_executor
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        with patch("builtins.print") as mock_print:
            project_analyzer.analyze_projects_parallel("mock_base", 1)
            
            # Check print args
            found = any("Found 3 code smells" in str(c) for c in mock_print.call_args_list)
            assert found


def test_save_results_non_empty_writes_file(monkeypatch, project_analyzer, tmp_path):
    """
    Test that _save_results actually writes to a file when df is not empty.
    NO MOCK on _save_results itself.
    """
    # Create a real dataframe
    df = pd.DataFrame({"col": [1, 2]})
    
    # Check that output path is set to tmp_path (via fixture)
    assert project_analyzer.output_path == str(tmp_path)
    
    project_analyzer._save_results(df, "test_real_save.csv")
    
    expected_file = tmp_path / "test_real_save.csv"
    assert expected_file.exists()
    
    # Verify content
    saved_df = pd.read_csv(expected_file)
    assert len(saved_df) == 2


def test_analyze_project_no_smells(monkeypatch, project_analyzer, mock_file_related_methods, tmp_path):
    """
    Test analyze_project when 0 smells are found (coverage 85->87 branch false).
    """
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("components.project_analyzer.ProjectAnalyzer._save_results", lambda self, df, path: None)
    
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", lambda p: ["files.py"])
    
    # Return empty smells
    project_analyzer.inspector.inspect = MagicMock(return_value=pd.DataFrame(columns=["filename"]))
    
    with patch("builtins.print") as mock_print:
        project_analyzer.analyze_project("mock_path")
        
        # Verify NO "Found X code smells" print
        found = any("Found" in str(c) and "code smells in file" in str(c) for c in mock_print.call_args_list)
        assert not found


def test_analyze_projects_sequential_skips_output_dir(monkeypatch, project_analyzer):
    """
    Test that analyze_projects_sequential skips 'output' and 'execution_log.txt'.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["output", "execution_log.txt", "project1"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    
    # Only project1 should trigger inspect
    # Return valid DF to avoid concat error
    project_analyzer.inspector.inspect = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", lambda p: ["f.py"])
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda p: None)
    monkeypatch.setattr("utils.file_utils.FileUtils.append_to_log", lambda p, d: None)
    monkeypatch.setattr("components.project_analyzer.ProjectAnalyzer._save_results", lambda s, d, p: None)

    project_analyzer.analyze_projects_sequential("mock_base")
    
    # Should be called once for project1, not for output or log
    assert project_analyzer.inspector.inspect.call_count == 1


def test_analyze_projects_sequential_inner_exception(monkeypatch, project_analyzer, tmp_path):
    """
    Test handling of SyntaxError/FileNotFoundError inside the file loop (lines 166-171).
    """
    monkeypatch.setattr("os.listdir", lambda p: ["proj1"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", lambda p: ["bad.py"])
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda p: None)
    
    project_analyzer.inspector.inspect = MagicMock(side_effect=SyntaxError("Bad Syntax"))
    
    # Capture error file write
    error_file = tmp_path / "error.txt"
    
    project_analyzer.analyze_projects_sequential("mock_base")
    
    assert error_file.exists()
    content = error_file.read_text()
    assert "Error in file bad.py: Bad Syntax" in content


def test_analyze_projects_parallel_no_smells(monkeypatch, project_analyzer):
    """
    Test parallel analysis with 0 smells to cover the 'if smell_count > 0' false branch.
    """
    monkeypatch.setattr("os.listdir", lambda p: ["proj1"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", lambda p: ["f.py"])
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda p: None)
    
    project_analyzer.inspector.inspect = MagicMock(return_value=pd.DataFrame(columns=["filename"]))
    
    # Mock threading stuff
    monkeypatch.setattr("utils.file_utils.FileUtils.synchronized_append_to_log", lambda p, pr, l: None)
    monkeypatch.setattr("components.project_analyzer.ProjectAnalyzer._save_results", lambda s, d, p: None)

    with patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor:
        mock_executor = MagicMock()
        MockExecutor.return_value = mock_executor
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        with patch("builtins.print") as mock_print:
            project_analyzer.analyze_projects_parallel("mock_base", 1)
            # Ensure NO print about found smells
            found = any("Found" in str(c) and "code smells in file" in str(c) for c in mock_print.call_args_list)
            assert not found


def test_analyze_projects_parallel_outer_exception(monkeypatch, project_analyzer):
    """
    Test handling of exception in the parallel task (lines 275-276).
    """
    monkeypatch.setattr("os.listdir", lambda p: ["proj1"])
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("utils.file_utils.FileUtils.initialize_log", lambda p: None)
    
    # Raise exception before loop
    monkeypatch.setattr("utils.file_utils.FileUtils.get_python_files", MagicMock(side_effect=RuntimeError("Parallel Fail")))
    
    with patch("concurrent.futures.ThreadPoolExecutor") as MockExecutor:
        mock_executor = MagicMock()
        MockExecutor.return_value = mock_executor
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        with patch("builtins.print") as mock_print:
            project_analyzer.analyze_projects_parallel("mock_base", 1)
            
            found = any("Error analyzing project 'proj1': Parallel Fail" in str(c) for c in mock_print.call_args_list)
            assert found

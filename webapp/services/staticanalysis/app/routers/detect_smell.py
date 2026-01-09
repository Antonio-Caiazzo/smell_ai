import os
import tempfile
import zipfile
from pathlib import Path  # Added based on existing usage in the file

from fastapi import APIRouter, Body, File, Form, UploadFile

from webapp.services.staticanalysis.app.schemas.requests import DetectSmellRequest
from webapp.services.staticanalysis.app.schemas.responses import (
    DetectCallGraphResponse,
    DetectSmellStaticResponse,
)
from webapp.services.staticanalysis.app.utils.static_analysis import (
    detect_static,
    detect_static_with_graph,
)

router = APIRouter()


@router.post("/detect_smell_static", response_model=DetectSmellStaticResponse)
async def detect_smell_static(payload: DetectSmellRequest):
    code_snippet = payload.code_snippet
    analysis_result = detect_static(code_snippet)
    return DetectSmellStaticResponse(
        success=analysis_result["success"], smells=analysis_result["response"]
    )


@router.post("/detect_call_graph", response_model=DetectCallGraphResponse)
async def detect_call_graph(
    file: UploadFile = File(None),
    code_snippet: str = Form(None),
    include_call_graph: bool = Form(True),
):
    """
    Detects code smells and optionally generates a call graph.
    Accepts either a code_snippet string OR a file (single .py or .zip).
    """
    smells = []
    graph = None

    # Handle Uploaded File (ZIP or Single Python)
    if file:
        content = await file.read()

        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Check if ZIP
            if file.filename.endswith(".zip"):
                # Extract to a temp dir
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # Analyze the directory
                    result = detect_static_with_graph(temp_dir, is_directory=True)
                    smells = result.get("response", [])
                    graph = result.get("call_graph")

                    temp_dir_real = os.path.realpath(temp_dir)

                    if graph and "nodes" in graph:
                        for node in graph["nodes"]:
                            if "file_path" in node:
                                n_path = node["file_path"]
                                if n_path.startswith(temp_dir):
                                    node["file_path"] = os.path.relpath(
                                        n_path, temp_dir
                                    )
                                elif os.path.realpath(n_path).startswith(temp_dir_real):
                                    node["file_path"] = os.path.relpath(
                                        os.path.realpath(n_path), temp_dir_real
                                    )

                    for smell in smells:
                        f_key = "filename" if "filename" in smell else "file"

                        if f_key in smell and smell[f_key]:
                            original_path = smell[f_key]
                            original_path_real = os.path.realpath(original_path)

                            rel_path = None
                            if original_path.startswith(temp_dir):
                                rel_path = os.path.relpath(original_path, temp_dir)
                            elif original_path_real.startswith(temp_dir_real):
                                rel_path = os.path.relpath(
                                    original_path_real, temp_dir_real
                                )

                            if rel_path:
                                smell["file"] = rel_path
                                smell["filename"] = rel_path
                        else:
                            pass

                    if graph and "nodes" in graph:
                        for node in graph["nodes"]:
                            if "file_path" in node and node["file_path"].startswith(
                                temp_dir
                            ):
                                node["file_path"] = os.path.relpath(
                                    node["file_path"], temp_dir
                                )
            else:
                # Single File

                with open(tmp_path, "r") as f:
                    text_content = f.read()

                result = detect_static_with_graph(
                    text_content, file_path=tmp_path, original_filename=file.filename
                )
                smells = result.get("response", [])
                graph = result.get("call_graph")

                if graph and "nodes" in graph:
                    for node in graph["nodes"]:

                        if "file_path" in node:
                            if node["file_path"] == tmp_path:
                                node["file_path"] = file.filename
                            else:
                                node["file_path"] = os.path.basename(node["file_path"])

                # Inject file name into smells for single file results
                for smell in smells:
                    smell.file = file.filename
                    smell.filename = file.filename

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Handle Raw Code Snippet
    elif code_snippet:
        result = detect_static_with_graph(code_snippet)
        smells = result.get("response", [])
        graph = result.get("call_graph")

        if graph and "nodes" in graph:
            for node in graph["nodes"]:
                node["file_path"] = "snippet.py"

    else:
        return DetectCallGraphResponse(success=False, smells=None, call_graph=None)

    return DetectCallGraphResponse(success=True, smells=smells, call_graph=graph)

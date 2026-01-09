import ast
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class Node:
    id: str
    name: str
    module: str
    package: str
    type: str  # "function" or "method" or "class"
    file_path: str
    start_line: int
    end_line: int
    source_code: str = ""


@dataclass
class CallSite:
    file_path: str
    line: int
    snippet: str


@dataclass
class Edge:
    source: str
    target: str
    call_sites: List[CallSite] = field(default_factory=list)


@dataclass
class CallGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)


class CallGraphGenerator:
    """
    Generates a call graph for the analyzed project.
    """

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.nodes_map: Dict[str, Node] = {}  # id -> Node
        self.edges_map: Dict[tuple, Edge] = {}  # (source, target) -> Edge
        self.function_definitions: Dict[str, str] = (
            {}
        )  # partial_name -> full_id (simplified mapping)

    def generate(self, file_paths: List[str]) -> Dict:
        """
        Generates the call graph JSON structure.
        """
        # Pass 1: Collect definitions (Nodes)
        for file_path in file_paths:
            self._scan_definitions(file_path)

        # Pass 2: Collect calls (Edges)
        for file_path in file_paths:
            self._scan_calls(file_path)

        # Build output
        graph = CallGraph(
            nodes=list(self.nodes_map.values()), edges=list(self.edges_map.values())
        )
        return asdict(graph)

    def generate_dot(self, file_paths: List[str]) -> str:
        """
        Generates the call graph in DOT format.
        """
        # Ensure graph is built (re-using generate logic if needed, but assuming generate was called or we reuse the maps)
        # Actually generate() builds the maps. If we want to generate DOT, we should make sure we have the data.
        # But usually one calls generate() first. Or we can just use the internal maps if they are populated.
        # Let's assume the user calls generate() first or we can just run the scan if empty.

        if not self.nodes_map and not self.edges_map:
            self.generate(file_paths)

        dot_lines = ["digraph CallGraph {", "    node [shape=box];"]

        # Nodes
        for node in self.nodes_map.values():
            # Escape quotes in label
            label = f"{node.name}\\n({node.type})"
            dot_lines.append(f'    "{node.id}" [label="{label}"];')

        # Edges
        for edge in self.edges_map.values():
            # Add edge with count of calls if multiple
            label = ""
            if len(edge.call_sites) > 1:
                label = f' [label="{len(edge.call_sites)}"]'
            dot_lines.append(f'    "{edge.source}" -> "{edge.target}"{label};')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def _get_module_info(self, file_path: str):
        try:
            rel_path = os.path.relpath(file_path, self.project_root)
        except ValueError:
            # Fallback if file_path is not under project_root (e.g. temp file)
            rel_path = os.path.basename(file_path)

        module_path = os.path.splitext(rel_path)[0].replace(os.sep, ".")
        package = module_path.split(".")[0]
        return module_path, package

    def _get_line_snippet(self, file_path: str, line: int) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if 1 <= line <= len(lines):
                    return lines[line - 1].strip()
        except:
            pass
        return ""

    def _get_source_segment(self, file_path: str, start: int, end_line: int) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 1-based indexing
                if 1 <= start <= end_line <= len(lines):
                    return "".join(lines[start - 1 : end_line])
        except:
            pass
        return "Source not available"

    def _scan_definitions(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception:
            return

        module, package = self._get_module_info(file_path)

        class DefinitionVisitor(ast.NodeVisitor):
            def __init__(self, generator, module, package, file_path):
                self.generator = generator
                self.module = module
                self.package = package
                self.file_path = file_path
                self.scope_stack = []
                # Helper to determine if we are directly inside a class (for "method" type)
                self.in_class_stack = []

            def _get_current_id_prefix(self):
                # e.g. "my_module.MyClass"
                if not self.scope_stack:
                    return self.module
                return f"{self.module}.{'.'.join(self.scope_stack)}"

            def visit_FunctionDef(self, node):
                self._handle_function(node)

            def visit_AsyncFunctionDef(self, node):
                self._handle_function(node)

            def _handle_function(self, node):
                prefix = self._get_current_id_prefix()
                node_id = f"{prefix}.{node.name}"

                # Determine type
                is_method = self.in_class_stack[-1] if self.in_class_stack else False
                node_type = "method" if is_method else "function"

                self.generator._add_node(
                    node_id,
                    node.name,
                    self.module,
                    self.package,
                    node_type,
                    self.file_path,
                    node.lineno,
                    node.end_lineno,
                )

                # Simplified mapping (last definition wins for partial name)
                self.generator.function_definitions[node.name] = node_id
                # Also store full ID map
                self.generator.function_definitions[node_id] = node_id

                # Enter scope
                self.scope_stack.append(node.name)
                # Inside a function, we are NOT directly in a class anymore (even if nested)
                self.in_class_stack.append(False)

                self.generic_visit(node)

                # Exit scope
                self.in_class_stack.pop()
                self.scope_stack.pop()

            def visit_ClassDef(self, node):
                prefix = self._get_current_id_prefix()
                f"{prefix}.{node.name}"

                # We could add the class itself as a node if needed, but the original code
                # didn't seem to treat classes as semantic nodes in the call graph (only methods/functions).
                # However, for correct scoping, we must traverse it.

                # Enter scope
                self.scope_stack.append(node.name)
                self.in_class_stack.append(True)

                self.generic_visit(node)

                self.in_class_stack.pop()
                self.scope_stack.pop()

        visitor = DefinitionVisitor(self, module, package, file_path)
        visitor.visit(tree)

    def _add_node(
        self, id, name, module, package, type, file_path, start_line, end_line
    ):
        if id not in self.nodes_map:
            source = self._get_source_segment(file_path, start_line, end_line)
            # Default type if missing (should not happen with new logic)
            self.nodes_map[id] = Node(
                id, name, module, package, type, file_path, start_line, end_line, source
            )

    def _add_edge(self, source, target, file_path, line):
        # Allow self-loops as per standard call graph definitions
        key = (source, target)
        if key not in self.edges_map:
            self.edges_map[key] = Edge(source=source, target=target)

        snippet = self._get_line_snippet(file_path, line)
        self.edges_map[key].call_sites.append(
            CallSite(file_path=file_path, line=line, snippet=snippet)
        )

    def _scan_calls(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception:
            return

        module, package = self._get_module_info(file_path)

        class CallVisitor(ast.NodeVisitor):
            def __init__(self, generator, current_module):
                self.generator = generator
                self.current_module = current_module
                self.scope_stack = []

            def visit_FunctionDef(self, node):
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_ClassDef(self, node):
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_Call(self, node):
                if not self.scope_stack:
                    return

                # Determine source ID
                source_suffix = ".".join(self.scope_stack)
                source_id = f"{self.current_module}.{source_suffix}"

                # Resolving Target
                target_id = self._resolve_target(node)
                if target_id and target_id in self.generator.nodes_map:
                    self.generator._add_edge(
                        source_id, target_id, file_path, node.lineno
                    )

            def _resolve_target(self, node):
                # Case 1: simple function call func() or Class()
                if isinstance(node.func, ast.Name):
                    name = node.func.id

                    # 1a. Check exact match absolute path: module.name
                    candidate = f"{self.current_module}.{name}"
                    if candidate in self.generator.nodes_map:
                        return candidate

                    # 1b. Check suffix match (imported)
                    if name in self.generator.function_definitions:
                        return self.generator.function_definitions[name]

                    # 1c. Scan all nodes for suffixes (fallback)
                    for nid in self.generator.nodes_map:
                        if nid.endswith(f".{name}"):
                            return nid
                        if nid.endswith(f".{name}.__init__"):
                            return nid

                # Case 2: Attribute call obj.method()
                elif isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr

                    # 2a. Self.method()
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == "self":
                            # Try from nearest parent upwards
                            current_prefix = f"{self.current_module}.{'.'.join(self.scope_stack[:-1])}"  # Pop method name
                            candidate = f"{current_prefix}.{method_name}"
                            if candidate in self.generator.nodes_map:
                                return candidate

                            # Fallback: traverse up the stack?
                            # For now, just continue to other heuristics.

                    # 2b. Heuristic: Unique Method Name Analysis
                    matches = [
                        nid
                        for nid in self.generator.nodes_map
                        if nid.endswith(f".{method_name}")
                    ]
                    if len(matches) == 1:
                        return matches[0]

                    # 2c. If multiple matches, assume match if obj name matches class name part
                    if isinstance(node.func.value, ast.Name):
                        obj_name = node.func.value.id
                        # Global search for string end pattern "obj_name.method_name"
                        suffix = f"{obj_name}.{method_name}"
                        for nid in self.generator.nodes_map:
                            if nid.endswith(suffix):
                                return nid

                return None

        visitor = CallVisitor(self, module)
        visitor.visit(tree)

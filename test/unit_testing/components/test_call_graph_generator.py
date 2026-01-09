import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from components.call_graph_generator import CallGraphGenerator


class TestCallGraphGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.generator = CallGraphGenerator(self.test_dir)

        # Create dummy file structure
        self.file_structure = {
            "module_a.py": """
def func_a():
    pass
""",
            "module_b.py": """
from module_a import func_a

def func_b():
    func_a()
    
class MyClass:
    def method_c(self):
        func_b()

    def method_d(self):
        self.method_c()
""",
        }

        for filename, content in self.file_structure.items():
            path = os.path.join(self.test_dir, filename)
            with open(path, "w") as f:
                f.write(content)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_dummy_file(self, filename, content):
        path = os.path.join(self.test_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_generate_nodes_and_edges(self):
        files = [os.path.join(self.test_dir, f) for f in self.file_structure.keys()]

        graph = self.generator.generate(files)
        nodes = graph["nodes"]
        edges = graph["edges"]

        # Verify Nodes
        node_ids = {n["id"] for n in nodes}
        self.assertIn("module_a.func_a", node_ids)
        self.assertIn("module_b.func_b", node_ids)
        self.assertIn("module_b.MyClass.method_c", node_ids)
        self.assertIn("module_b.MyClass.method_d", node_ids)

        # Verify Edges
        edge_set = {(e["source"], e["target"]) for e in edges}
        self.assertIn(("module_b.func_b", "module_a.func_a"), edge_set)
        self.assertIn(("module_b.MyClass.method_c", "module_b.func_b"), edge_set)
        self.assertIn(
            ("module_b.MyClass.method_d", "module_b.MyClass.method_c"), edge_set
        )

    # --- generate_dot() tests ---

    def test_generate_dot_triggers_generate_when_empty(self):
        """Ensures generate is called if maps are empty."""
        filename = "simple.py"
        self.create_dummy_file(filename, "def foo(): pass")
        
        dot_output = self.generator.generate_dot([os.path.join(self.test_dir, filename)])
        
        self.assertIn("digraph CallGraph", dot_output)
        self.assertIn("simple.foo", dot_output)

    def test_generate_dot_does_not_regenerate_when_maps_present(self):
        """Ensures generate is NOT called if maps are populated."""
        filename = "simple2.py"
        path = self.create_dummy_file(filename, "def bar(): pass")
        
        # Populate maps
        self.generator.generate([path])
        
        with patch.object(self.generator, 'generate') as mock_gen:
            dot_output = self.generator.generate_dot([path])
            mock_gen.assert_not_called()
        
        self.assertIn("simple2.bar", dot_output)

    def test_generate_dot_edge_label_when_multiple_calls(self):
        """Checks edge label for multiple calls."""
        content = """
def callee(): pass
def caller():
    callee()
    callee()
"""
        path = self.create_dummy_file("multi_calls.py", content)
        self.generator.generate([path])
        dot = self.generator.generate_dot([path])
        
        self.assertIn('-> "multi_calls.callee" [label="2"]', dot)

    def test_generate_dot_single_call_edge(self):
        """Checks edge without label for single call."""
        path = self.create_dummy_file("single_dot.py", "def a(): pass\ndef b(): a()")
        self.generator.generate([path])
        dot = self.generator.generate_dot([path])
        self.assertIn('"single_dot.b" -> "single_dot.a";', dot)
        self.assertNotIn('label="1"', dot)

    # --- _get_module_info() tests ---

    @patch('os.path.relpath')
    def test_get_module_info_relpath_valueerror_fallbacks_to_basename(self, mock_relpath):
        """Fallback to basename on ValueError from relpath."""
        mock_relpath.side_effect = ValueError("Boom")
        
        mod, pkg = self.generator._get_module_info("/tmp/somefile.py")
        
        self.assertEqual(mod, "somefile")
        self.assertEqual(pkg, "somefile")

    # --- Snippet extraction tests ---

    def test_get_line_snippet_valid_line(self):
        content = """line1
line2
line3"""
        path = self.create_dummy_file("snippet.py", content)
        snippet = self.generator._get_line_snippet(path, 2)
        self.assertEqual(snippet, "line2")

    def test_get_line_snippet_out_of_range(self):
        path = self.create_dummy_file("snippet_out.py", "line1")
        snippet = self.generator._get_line_snippet(path, 999)
        self.assertEqual(snippet, "")

    def test_get_line_snippet_open_exception_returns_empty(self):
        with patch('builtins.open', side_effect=OSError):
            snippet = self.generator._get_line_snippet("any_path", 1)
            self.assertEqual(snippet, "")

    def test_get_source_segment_valid_range(self):
        content = """A
B
C
D"""
        path = self.create_dummy_file("source.py", content)
        # Lines 2 and 3
        segment = self.generator._get_source_segment(path, 2, 3)
        self.assertEqual(segment, "B\nC\n")

    def test_get_source_segment_invalid_range_returns_fallback(self):
        path = self.create_dummy_file("source_inv.py", "A")
        segment = self.generator._get_source_segment(path, 5, 6)
        self.assertEqual(segment, "Source not available")

    def test_get_source_segment_open_exception_returns_fallback(self):
        with patch('builtins.open', side_effect=OSError):
            segment = self.generator._get_source_segment("any", 1, 2)
            self.assertEqual(segment, "Source not available")

    # --- Robustness tests (syntax errors, existing nodes) ---

    def test_scan_definitions_syntax_error_is_ignored(self):
        path = self.create_dummy_file("bad_syntax.py", "this is not python code")
        self.generator._scan_definitions(path)

    def test_scan_calls_syntax_error_is_ignored(self):
        path = self.create_dummy_file("bad_syntax_calls.py", "this is not python code")
        self.generator._scan_calls(path)

    def test_top_level_call_is_ignored(self):
        content = """
def f(): pass
f()  # top-level call
"""
        path = self.create_dummy_file("toplevel.py", content)
        self.generator.generate([path])
        self.assertEqual(len(self.generator.edges_map), 0)

    def test_add_node_existing_key_idempotence(self):
        """Verifies _add_node does not overwrite existing node."""
        path = self.create_dummy_file("manual_node.py", "")
        self.generator._add_node("id1", "n", "m", "p", "func", path, 1, 1)
        obj1 = self.generator.nodes_map["id1"]
        
        # Try to add same ID again
        self.generator._add_node("id1", "n_new", "m", "p", "func", path, 1, 1)
        obj2 = self.generator.nodes_map["id1"]
        
        self.assertIs(obj1, obj2)

    def test_add_edge_existing_key_increments_call_sites(self):
        """Verifies _add_edge appends to call_sites for existing edge."""
        path = self.create_dummy_file("manual_edge.py", "")
        self.generator._add_edge("s", "t", path, 10)
        edge1 = self.generator.edges_map[("s", "t")]
        self.assertEqual(len(edge1.call_sites), 1)
        
        # Add same edge again
        self.generator._add_edge("s", "t", path, 20)
        edge2 = self.generator.edges_map[("s", "t")]
        
        self.assertIs(edge1, edge2)
        self.assertEqual(len(edge1.call_sites), 2)

    # --- Target Resolution Tests ---

    def test_async_functions(self):
        content = """
async def afunc(): pass
async def acaller():
    await afunc()
"""
        path = self.create_dummy_file("async_test.py", content)
        graph = self.generator.generate([path])
        self.assertIn("async_test.afunc", {n['id'] for n in graph['nodes']})
        self.assertIn(("async_test.acaller", "async_test.afunc"), {(e['source'], e['target']) for e in graph['edges']})

    def test_resolve_target_case_1a_same_module_function(self):
        content = """
def helper(): pass
def caller(): helper()
"""
        path = self.create_dummy_file("same_mod.py", content)
        self.generator.generate([path])
        self.assertIn(("same_mod.caller", "same_mod.helper"), self.generator.edges_map)

    def test_resolve_target_case_1b_explicit(self):
        """Resolves target via imported definition."""
        pathA = self.create_dummy_file("mod_def.py", "def mytarget(): pass")
        pathB = self.create_dummy_file("mod_use.py", "from mod_def import mytarget\ndef user(): mytarget()")
        
        self.generator.generate([pathA, pathB])
        self.assertIn(("mod_use.user", "mod_def.mytarget"), self.generator.edges_map)

    def test_resolve_target_case_1c_suffix_fallback(self):
        """Resolves target via suffix fallback when import mapping is missing."""
        pathA = self.create_dummy_file("mod_loop.py", "def loopfunc(): pass")
        pathB = self.create_dummy_file("mod_call.py", "def caller(): loopfunc()")
        
        # Intercept to clear definitions
        self.generator._scan_definitions(pathA)
        self.generator._scan_definitions(pathB)
        self.generator.function_definitions = {} 
        
        self.generator._scan_calls(pathA)
        self.generator._scan_calls(pathB)
        
        self.assertIn(("mod_call.caller", "mod_loop.loopfunc"), self.generator.edges_map)

    def test_resolve_target_class_instantiation_maps_to_init(self):
        content = """
class MyClass:
    def __init__(self): pass

def creator():
    MyClass()
"""
        path = self.create_dummy_file("init_test.py", content)
        self.generator.generate([path])
        self.assertIn(("init_test.creator", "init_test.MyClass.__init__"), self.generator.edges_map)

    def test_complex_call_expression_ignored(self):
        """Ignores complex call expressions (e.g. lambdas)."""
        path = self.create_dummy_file("complex_expr.py", "def f(): (lambda:0)()")
        self.generator.generate([path])
        sources = {e.source for e in self.generator.edges_map.values()}
        self.assertNotIn("complex_expr.f", sources)

    def test_resolve_target_attribute_unique_method(self):
        content = """
class A:
    def unique_one(self): pass

def caller():
    obj = A()
    obj.unique_one()
"""
        path = self.create_dummy_file("unique.py", content)
        self.generator.generate([path])
        self.assertIn(("unique.caller", "unique.A.unique_one"), self.generator.edges_map)

    def test_resolve_target_attribute_multiple_matches_objname_selects_correct(self):
        content = """
class Alpha:
    def run(self): pass

class Beta:
    def run(self): pass

def main():
    Alpha = None
    Alpha.run()
"""
        path = self.create_dummy_file("obj_match.py", content)
        self.generator.generate([path])
        self.assertIn(("obj_match.main", "obj_match.Alpha.run"), self.generator.edges_map)
        self.assertNotIn(("obj_match.main", "obj_match.Beta.run"), self.generator.edges_map)

    def test_attribute_call_on_non_name(self):
        """Resolves unique method on complex object expression."""
        path = self.create_dummy_file("attr_noname.py", """
class K:
    def unique_m(self): pass
def f():
    get_K().unique_m()
""")
        self.generator.generate([path])
        self.assertIn(("attr_noname.f", "attr_noname.K.unique_m"), self.generator.edges_map)

    def test_resolve_self_method_parent_lookup(self):
        path = self.create_dummy_file("self_lookup.py", """
class P:
    def parent_m(self): pass
    def child_m(self): self.parent_m()
""")
        self.generator.generate([path])
        self.assertIn(("self_lookup.P.child_m", "self_lookup.P.parent_m"), self.generator.edges_map)

    def test_resolve_ambiguous_attribute(self):
        """Resolves ambiguous attribute on complex object (returns None)."""
        path = self.create_dummy_file("ambig_fail.py", """
class X:
    def run(self): pass
class Y:
    def run(self): pass
def f():
    get_obj().run()
""")
        self.generator.generate([path])
        sources = {e.source for e in self.generator.edges_map.values()}
        self.assertNotIn("ambig_fail.f", sources)

    # --- Negative tests ---

    def test_resolve_target_returns_none_for_unknown(self):
        content = """
def caller():
    unknown_function()
"""
        path = self.create_dummy_file("unknown.py", content)
        self.generator.generate([path])
        self.assertEqual(len(self.generator.edges_map), 0)

    def test_resolve_self_method_missing(self):
        path = self.create_dummy_file("self_miss.py", """
class M:
    def foo(self):
        self.missing_method()
""")
        self.generator.generate([path])
        sources = {e.source for e in self.generator.edges_map.values()}
        self.assertNotIn("self_miss.M.foo", sources)

    def test_resolve_ambiguous_method_no_match(self):
        path = self.create_dummy_file("ambig_loop_miss.py", """
class A:
    def common(self): pass
class B:
    def common(self): pass

def f():
    C = A()
    C.common()
""")
        self.generator.generate([path])
        edge_key = ("ambig_loop_miss.f", "ambig_loop_miss.C.common") 
        self.assertNotIn(edge_key, self.generator.edges_map)


if __name__ == "__main__":
    unittest.main()

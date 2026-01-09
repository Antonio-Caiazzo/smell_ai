from fastapi.testclient import TestClient

from webapp.gateway import main

# flake8: noqa

# Create the test client
client = TestClient(main.app)


def test_gateway_to_call_graph_snippet():
    code_snippet = """
def foo():
    pass

def bar():
    foo()
"""
    data = {
        "code_snippet": code_snippet,
        "include_call_graph": True
    }
    
    # helper to simulate form data if needed, but TestClient handles json vs data
    # The endpoint expects Form data for code_snippet if file is not provided
    response = client.post("/api/detect_call_graph", data=data)

    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["success"] is True
    assert "call_graph" in json_resp
    assert json_resp["call_graph"] is not None
    
    # Check basic graph structure
    nodes = json_resp["call_graph"].get("nodes", [])
    edges = json_resp["call_graph"].get("edges", [])
    
    # We expect at least nodes for foo and bar
    node_names = [n.get("name") for n in nodes]
    assert "foo" in node_names or "snippet.foo" in node_names
    assert "bar" in node_names or "snippet.bar" in node_names
    
    # We expect an edge from bar to foo
    assert len(edges) > 0


def test_gateway_to_call_graph_no_input():
    response = client.post("/api/detect_call_graph", data={})
    # Depending on implementation, might be 200 with success=False or 422
    # The current implementation returns success=False if no input
    assert response.status_code == 200
    assert response.json()["success"] is False

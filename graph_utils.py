import webbrowser

def save_mermaid_diagram(graph, filename="docs/graph.mmd"):
    """
    Saves a LangGraph graph as a Mermaid (.mmd) file and opens it in mermaid.live.
    
    Args:
        graph: A LangGraph graph object.
        filename (str): Name of the file to save (default is 'graph.mmd').
    """
    try:
        mermaid_code = graph.draw_mermaid()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print(f"✅ Mermaid diagram saved to: {filename}")
        
        print("🌐 Opening https://mermaid.live - paste your diagram code there.")
        webbrowser.open("https://mermaid.live")

    except Exception as e:
        print(f"❌ Failed to save Mermaid diagram: {e}")
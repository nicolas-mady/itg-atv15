import networkx as nx
import matplotlib.pyplot as plt
import os

os.makedirs("../imgs", exist_ok=True)

def simular_microsoft_graphrag_comunidades():
    """
    Simula a fase de indexação do Microsoft GraphRAG baseada em comunidades.
    Referência: "Utiliza detecção de comunidades... para criar resumos hierárquicos." 
    """
    print("--- 1. Algoritmo de Detecção de Comunidades (Estilo Microsoft GraphRAG) ---")
    
    # Criar um grafo de exemplo (simulando entidades e relações extraídas de documentos)
    G = nx.erdos_renyi_graph(n=20, p=0.2, seed=42)
    
    # Adicionar atributos de texto simulados aos nós (Conceito de TAG - Textual Attribute Graph) 
    for i in G.nodes():
        G.nodes[i]['texto'] = f"Entidade_{i}"

    # Aplicar detecção de comunidades (Algoritmo de Modularidade Gulosa como proxy para Leiden)
    comunidades = nx.community.greedy_modularity_communities(G)
    
    # Visualização
    pos = nx.spring_layout(G, seed=42)
    cores = ['r', 'b', 'g', 'y', 'c', 'm']
    plt.figure(figsize=(8, 6))
    
    print(f"Número de comunidades detectadas: {len(comunidades)}")
    
    for i, comunidade in enumerate(comunidades):
        lista_nos = list(comunidade)
        cor = cores[i % len(cores)]
        nx.draw_networkx_nodes(G, pos, nodelist=lista_nos, node_color=cor, label=f"Comunidade {i}")
        
        # Simula o "Map-Reduce" gerando um resumo por comunidade 
        print(f"Comunidade {i}: Nós {lista_nos} -> [LLM geraria resumo aqui]")

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title("Detecção de Comunidades (Hierarquia para Resumos)")
    plt.legend()
    
    # Save the plot instead of showing it interactively
    plt.savefig("../imgs/community_detection_graph.png", dpi=300, bbox_inches='tight')
    print("Gráfico salvo como '../imgs/community_detection_graph.png'")
    plt.close()  # Close the figure to free memory

# Executar
if __name__ == "__main__":
    simular_microsoft_graphrag_comunidades()
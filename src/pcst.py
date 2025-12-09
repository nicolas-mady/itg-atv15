import networkx as nx
import matplotlib.pyplot as plt
import os

os.makedirs("../imgs", exist_ok=True)

def heuristica_pcst_g_retriever(G, query_nodes, budget_custo=5.0):
    """
    Simula a recuperação baseada em PCST (Prize-Collecting Steiner Tree).
    Referência: "Busca um subgrafo conectado que maximize prêmios e minimize custos" .
    
    :param G: O Grafo de conhecimento
    :param query_nodes: Nós identificados como relevantes para a pergunta (têm 'prêmios' altos)
    :param budget_custo: Custo máximo permitido para conectar os tópicos
    """
    print("\n--- 2. Algoritmo PCST Heurístico (Estilo G-Retriever) ---")
    
    # Subgrafo resultante inicial (apenas os nós de consulta)
    subgrafo_nos = set(query_nodes)
    subgrafo_arestas = []
    
    # Tentar conectar os nós de consulta usando caminhos de menor custo (Dijkstra)
    # Isso simula a busca por conexões semânticas fortes (baixo custo) entre tópicos relevantes
    pares_conectados = []
    sorted_query_nodes = list(query_nodes)
    
    for i in range(len(sorted_query_nodes)):
        for j in range(i + 1, len(sorted_query_nodes)):
            origem = sorted_query_nodes[i]
            destino = sorted_query_nodes[j]
            
            try:
                # Encontrar menor caminho ponderado pelo 'cost' da aresta
                caminho = nx.shortest_path(G, source=origem, target=destino, weight='cost')
                custo_caminho = nx.shortest_path_length(G, source=origem, target=destino, weight='cost')
                
                if custo_caminho <= budget_custo:
                    print(f"Conectando '{origem}' e '{destino}' (Custo: {custo_caminho:.2f}): {caminho}")
                    subgrafo_nos.update(caminho)
                    # Adicionar arestas do caminho
                    for k in range(len(caminho)-1):
                        subgrafo_arestas.append((caminho[k], caminho[k+1]))
                else:
                    print(f"Conexão '{origem}' -> '{destino}' muito custosa/distante ({custo_caminho:.2f}). Ignorada para reduzir ruído.")
                    
            except nx.NetworkXNoPath:
                pass

    return G.edge_subgraph(subgrafo_arestas).copy()

def executar_demo_pcst():
    # 1. Construir um Grafo com "Custos" nas arestas
    # Custo baixo = Forte relação semântica
    # Custo alto = Relação fraca
    G = nx.Graph()
    
    # Adicionando arestas com custos (weights)
    arestas = [
        ("Brasil", "Amazonas", 0.5), ("Amazonas", "Manaus", 0.2), 
        ("Manaus", "UFAM", 0.1), ("UFAM", "Pesquisa", 0.8),
        ("Pesquisa", "IA", 0.5), ("IA", "GraphRAG", 0.3),
        ("GraphRAG", "LLMs", 0.4), ("LLMs", "Alucinação", 0.3),
        ("Brasil", "Futebol", 0.1), ("Futebol", "Copa", 0.2), # Tópico irrelevante para a query técnica
        ("GraphRAG", "Microsoft", 0.2), ("Microsoft", "Windows", 0.9)
    ]
    G.add_weighted_edges_from(arestas, weight='cost')
    
    # 2. Definir "Prêmios" (Nós relevantes para uma query hipotética: "Como GraphRAG resolve alucinações?")
    # O G-Retriever atribuiria prêmios altos a nós que dão match com a query
    nos_relevantes = ["GraphRAG", "Alucinação", "UFAM"] 
    
    # 3. Executar a recuperação otimizada
    subgrafo = heuristica_pcst_g_retriever(G, nos_relevantes, budget_custo=2.0)
    
    # Visualização
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=10)
    
    # Desenhar grafo original em cinza claro
    nx.draw_networkx(G, pos, node_color='lightgray', edge_color='lightgray', with_labels=True, node_size=500)
    
    # Desenhar subgrafo recuperado em destaque (vermelho)
    nx.draw_networkx(subgrafo, pos, node_color='orange', edge_color='red', width=2, with_labels=True)
    
    plt.title("Recuperação Otimizada via PCST (G-Retriever)")
    plt.savefig("../imgs/pcst_retrieval_graph.png", dpi=300, bbox_inches='tight')
    print("Gráfico salvo como '../imgs/pcst_retrieval_graph.png'")
    plt.close()

if __name__ == "__main__":
    executar_demo_pcst()
import networkx as nx
import numpy as np

def simulacao_g_rag_reranker():
    """
    Simula o G-RAG: Melhora a lista de documentos usando conexões de grafo.
    Referência: "Identifica documentos relevantes por associação... conexões fracas com a consulta, 
    mas fortes com outros documentos já relevantes"[cite: 40].
    """
    print("--- 1. G-RAG: Reranker Associativo (Duplo Grafo) ---")

    # 1. Simulação da Busca Vetorial Inicial (Score de Similaridade com a Query)
    # Suponha que a query seja sobre "GraphRAG".
    # Doc A (Alta relevância), Doc B (Média), Doc C (Baixa - o vetor falhou em captar), Doc D (Irrelevante)
    docs_iniciais = {
        "Doc_A": 0.95, # Muito relevante
        "Doc_B": 0.60,
        "Doc_C": 0.20, # Falso negativo na busca vetorial (score baixo)
        "Doc_D": 0.10  # Irrelevante
    }
    
    print("1. Ranking Vetorial Inicial (Query <-> Documento):")
    for doc, score in sorted(docs_iniciais.items(), key=lambda x: x[1], reverse=True):
        print(f"   {doc}: {score:.2f}")

    # 2. Construção do Grafo de Documentos (Simulando conexões AMR/Citações) [cite: 141]
    # Doc A e Doc C discutem o mesmo conceito profundo, logo têm uma aresta forte.
    G = nx.Graph()
    G.add_edge("Doc_A", "Doc_C", weight=0.9) # Conexão forte "salva" o Doc C
    G.add_edge("Doc_B", "Doc_D", weight=0.3) # Conexão fraca
    G.add_edge("Doc_A", "Doc_B", weight=0.4) 

    # 3. Algoritmo de Reranking do G-RAG
    # Novo Score = (alpha * Score Vetorial) + (beta * Score dos Vizinhos Relevantes)
    alpha = 0.6
    beta = 0.4
    
    novos_scores = {}
    
    for doc in docs_iniciais:
        score_vizinhanca = 0
        if doc in G:
            vizinhos = G[doc]
            # Soma ponderada dos scores iniciais dos vizinhos * força da conexão
            for vizinho, dados_aresta in vizinhos.items():
                peso_aresta = dados_aresta['weight']
                score_vizinho = docs_iniciais.get(vizinho, 0)
                score_vizinhanca += (score_vizinho * peso_aresta)
        
        # Fórmula de combinação
        score_final = (alpha * docs_iniciais[doc]) + (beta * score_vizinhanca)
        novos_scores[doc] = score_final

    print("\n2. Ranking Final G-RAG (Considerando Associações):")
    ranking_final = sorted(novos_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (doc, score) in enumerate(ranking_final):
        status = "SUBIU" if score > docs_iniciais[doc] else "DESCEU/MANTEVE"
        print(f"   {i+1}º {doc}: {score:.2f} ({status})")
        if doc == "Doc_C":
            print(f"   -> Nota: Doc_C foi promovido devido à conexão forte com Doc_A (Lógica G-RAG) [cite: 40]")

if __name__ == "__main__":
    simulacao_g_rag_reranker()
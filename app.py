from flask import Flask, request, jsonify, send_file
import joblib
import networkx as nx
from model import GCN
import matplotlib.pyplot as plt
from io import BytesIO
#installation 
#pip install torch torchvision torchaudio
#pip install torch-geometric
# pip install networkx
#pip install joblib
#pip install matplotlib
def linear_threshold(G, initial_infected, node_thresholds):
    # Initialiser les ensembles de nœuds infectés
    newly_infected = set(initial_infected)  # Nœuds récemment infectés
    total_infected = set(initial_infected)  # Tous les nœuds infectés
    influence = {node: 0.0 for node in G.nodes}  # Influence exercée sur chaque nœud

    # Boucle jusqu'à ce qu'il n'y ait plus de propagation
    while newly_infected:
        next_infected = set()  # Liste des nœuds à infecter au prochain tour
        
        # Pour chaque nœud récemment infecté
        for node in newly_infected:
            # Examiner tous les voisins
            for neighbor in G.neighbors(node):
                if neighbor not in total_infected:  # Si le voisin n'est pas déjà infecté
                    # Ajouter l'influence basée sur le poids de l'arête
                    influence[neighbor] += G[node][neighbor]['weight']
                    
                    # Vérifier si le voisin a un seuil dans node_thresholds
                    if neighbor in node_thresholds:
                        # Si l'influence atteint ou dépasse le seuil du voisin, il devient infecté
                        if influence[neighbor] >= node_thresholds[neighbor]:
                            next_infected.add(neighbor)
                    else:
                        # Si le seuil du voisin est manquant, retourner une erreur ou utiliser un seuil par défaut
                        print(f"Warning: No threshold for neighbor {neighbor}, using default threshold.")
                        default_threshold = 0.5  # Ex: seuil par défaut
                        if influence[neighbor] >= default_threshold:
                            next_infected.add(neighbor)

        # Mettre à jour les ensembles de nœuds infectés
        total_infected.update(next_infected)
        newly_infected = next_infected  # Nouveaux nœuds infectés pour le prochain tour

    # Retourner la liste de tous les nœuds infectés
    return total_infected



app = Flask(__name__)

# Charger le modèle et la fonction
model_data = joblib.load('model_menace.pkl')
model = model_data['model']
linear_threshold = model_data['linear_threshold']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Charger les données du graphe
    edges = data['edges']
    initial_infected = data['initial_infected']
    node_thresholds = data['node_thresholds']

    # Construire le graphe
    G = nx.Graph()
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)

    # Appliquer la fonction linear_threshold
    infected_nodes = linear_threshold(G, initial_infected, node_thresholds)

    return jsonify({'infected_nodes': list(infected_nodes)})
@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.get_json()

    # Charger les données du graphe
    edges = data['edges']
    initial_infected = data['initial_infected']
    node_thresholds = data['node_thresholds']

    # Construire le graphe
    G = nx.Graph()
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)

    # Appliquer la fonction linear_threshold
    infected_nodes = linear_threshold(G, initial_infected, node_thresholds)

    # Positionner les nœuds
    pos = nx.spring_layout(G)

    # Créer les couleurs pour les nœuds
    node_colors = ['red' if node in infected_nodes else 'skyblue' for node in G.nodes]

    # Dessiner le graphe
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color='gray',
        node_size=2000,
        font_size=12,
        font_weight='bold'
    )
    plt.title("Propagation des menaces dans le réseau")

    # Sauvegarder le graphe dans un buffer
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    return send_file(img_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True,port=8000)

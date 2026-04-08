import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.integrate import quad

# Fonction de calcul de la probabilité pour un processus de Poisson non homogène
def calculate_probability_N(lambda_t_func, t_start, t_end, k):
    # Calcul du taux moyen d'événements (λt)
    rate_integral, _ = quad(lambda t: lambda_t_func(t), t_start, t_end)
    mean_rate = rate_integral
    # Calcul de la probabilité pour k événements
    probability = (mean_rate ** k) * np.exp(-mean_rate) / np.math.factorial(k)
    return probability


# Fonction pour simuler un processus de Markov
def simulate_markov(transition_matrix, initial_state, num_steps):
    current_state = initial_state
    states_history = [current_state]
    
    for _ in range(num_steps):
        # Sélectionner l'état suivant à partir de la matrice de transition
        current_state = np.random.choice(len(transition_matrix), p=transition_matrix[current_state])
        states_history.append(current_state)
        
    return states_history

# Fonction pour afficher la simulation sous forme de graphe avec NetworkX
def plot_simulation_markov(states_history, state_labels):
    """Affiche le graphe de transition des états avec NetworkX."""
    
    # Créer un graph orienté (diGraph)
    G = nx.DiGraph()

    # Ajouter les nœuds avec les états
    for i in range(len(state_labels)):
        G.add_node(i, label=state_labels[i])

    # Ajouter les arêtes en fonction de l'historique de simulation
    for i in range(len(states_history) - 1):
        start_state = states_history[i]
        end_state = states_history[i + 1]
        G.add_edge(start_state, end_state)
    
    # Dessiner le graphe avec Matplotlib
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # Position stable pour les nœuds

    # Dessiner les nœuds et les arêtes
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight='bold', edge_color="gray")
    
    # Ajouter les poids des arêtes (probabilités de transition)
    edge_labels = {}
    for (u, v) in G.edges():
        edge_labels[(u, v)] = f"{np.round(np.random.rand(), 2)}"  # On ajoute une probabilité aléatoire comme exemple
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Afficher avec Streamlit
    st.pyplot(plt)

# Fonction pour vérifier la validité de la matrice de transition
def is_valid_transition_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return False  # La matrice doit être carrée
    if np.any(matrix < 0):  # Vérifie que toutes les probabilités sont positives
        return False
    if np.any(np.abs(np.sum(matrix, axis=1) - 1) > 1e-6):  # Vérifie que la somme de chaque ligne est égale à 1
        return False
    return True

# Fonction pour générer une matrice de transition selon la distribution choisie
def generate_transition_matrix(num_states, distribution_type):
    if distribution_type == "Distribution uniforme":
        # Générer une matrice de transition avec des valeurs uniformément distribuées
        transition_matrix = np.random.rand(num_states, num_states)
        # Normaliser chaque ligne pour que la somme soit égale à 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    elif distribution_type == "Distribution de Dirichlet":
        # Générer une matrice de transition avec une distribution de Dirichlet
        transition_matrix = np.random.dirichlet(np.ones(num_states), size=num_states)
    elif distribution_type == "Distribution normale":
        # Générer une matrice de transition avec des valeurs de distribution normale
        transition_matrix = np.random.normal(0.5, 0.1, size=(num_states, num_states))
        # Normaliser chaque ligne pour que la somme soit égale à 1 et que les valeurs soient positives
        transition_matrix = np.abs(transition_matrix)  # Prendre les valeurs absolues pour éviter les négatifs
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Distribution non prise en charge")
    
    return transition_matrix

# Interface Streamlit pour la simulation du processus de Markov
def process_markov_simulation():
    st.subheader("Processus de Markov à temps discret")
    
    # Nombre d'états
    num_states = st.number_input("Nombre d'états", min_value=2, step=1, value=3)
    
    # Choisir la méthode pour définir la matrice de transition
    matrix_option = st.radio("Choisissez comment définir la matrice de transition", ["Générer aléatoirement", "Saisir manuellement"])
    
    # Initialiser la matrice de transition
    transition_matrix = []
    
    if matrix_option == "Générer aléatoirement":
        # Choisir la distribution pour générer la matrice de transition
        distribution_type = st.selectbox("Choisissez la distribution pour générer la matrice de transition", 
                                        ["Distribution uniforme", "Distribution de Dirichlet", "Distribution normale"])
        
        # Générer la matrice de transition
        transition_matrix = generate_transition_matrix(num_states, distribution_type)
        st.subheader("Matrice de Transition Générée")
        st.write(transition_matrix)
        
    elif matrix_option == "Saisir manuellement":
        st.subheader("Matrice de Transition Manuelle")
        for i in range(num_states):
            row = st.text_input(f"Probabilités de transition depuis l'état {i}", value="0.5, 0.5", help="Exemple : '0.5, 0.5' pour une matrice 2x2.")
            try:
                transition_matrix.append(list(map(float, row.split(','))))
            except ValueError:
                st.warning(f"Les valeurs de la ligne {i} doivent être des nombres séparés par des virgules.")
        
        transition_matrix = np.array(transition_matrix)

        # Vérification si chaque ligne contient bien 'num_states' éléments
        if transition_matrix.shape[1] != num_states:
            st.warning(f"Chaque ligne de la matrice doit avoir {num_states} éléments.")
            transition_matrix = []

    # Vérification de la validité de la matrice de transition
    if is_valid_transition_matrix(transition_matrix):
        # Choisir l'état initial
        initial_state = st.selectbox("Choisissez l'état initial", range(num_states))
        
        # Nombre d'étapes de la simulation
        num_steps = st.number_input("Nombre d'étapes de simulation", min_value=1, step=1, value=10)
        
        # Simuler le processus de Markov
        if st.button("Lancer la simulation"):
            states_history = simulate_markov(transition_matrix, initial_state, num_steps)
            state_labels = [f"État {i}" for i in range(num_states)]
            plot_simulation_markov(states_history, state_labels)
    else:
        st.error("La matrice de transition doit être valide : toutes les probabilités doivent être positives et la somme de chaque ligne doit être égale à 1.")

# Fonction pour tracer la simulation du processus de Poisson non homogène
def plot_simulation(lambda_t_func, t_start, t_end):
    t_values = np.linspace(t_start, t_end, 100)
    event_rates = [lambda_t_func(t) for t in t_values]
    
    # Tracer le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_values, event_rates, color="royalblue", linestyle='-', marker='o', markersize=5)
    ax.set_title("Simulation du Processus de Poisson Non Homogène", fontsize=18, color="darkblue")
    ax.set_xlabel("Temps (t)", fontsize=14, color="darkslategray")
    ax.set_ylabel("Taux d'événements λ(t)", fontsize=14, color="darkslategray")
    ax.grid(True)
    
    st.pyplot(fig)


# Interface Streamlit pour la simulation du processus de Markov
def process_markov_simulation():
    st.subheader("Processus de Markov à temps discret")
    
    # Nombre d'états
    num_states = st.number_input("Nombre d'états", min_value=2, step=1, value=3)
    
    # Choisir la méthode pour définir la matrice de transition
    matrix_option = st.radio("Choisissez comment définir la matrice de transition", ["Générer aléatoirement", "Saisir manuellement"])
    
    # Initialiser la matrice de transition
    transition_matrix = []
    
    if matrix_option == "Générer aléatoirement":
        # Choisir la distribution pour générer la matrice de transition
        distribution_type = st.selectbox("Choisissez la distribution pour générer la matrice de transition", 
                                        ["Distribution uniforme", "Distribution de Dirichlet", "Distribution normale"])
        
        # Générer la matrice de transition
        transition_matrix = generate_transition_matrix(num_states, distribution_type)
        st.subheader("Matrice de Transition Générée")
        st.write(transition_matrix)
        
    elif matrix_option == "Saisir manuellement":
        st.subheader("Matrice de Transition Manuelle")
        for i in range(num_states):
            row = st.text_input(f"Probabilités de transition depuis l'état {i}", value="0.5, 0.5", help="Exemple : '0.5, 0.5' pour une matrice 2x2.")
            try:
                transition_matrix.append(list(map(float, row.split(','))))
            except ValueError:
                st.warning(f"Les valeurs de la ligne {i} doivent être des nombres séparés par des virgules.")
        
        transition_matrix = np.array(transition_matrix)

        # Vérification si chaque ligne contient bien 'num_states' éléments
        if transition_matrix.shape[1] != num_states:
            st.warning(f"Chaque ligne de la matrice doit avoir {num_states} éléments.")
            transition_matrix = []

    # Vérification de la validité de la matrice de transition
    if is_valid_transition_matrix(transition_matrix):
        # Choisir l'état initial
        initial_state = st.selectbox("Choisissez l'état initial", range(num_states))
        
        # Nombre d'étapes de la simulation
        num_steps = st.number_input("Nombre d'étapes de simulation", min_value=1, step=1, value=10)
        
        # Simuler le processus de Markov
        if st.button("Lancer la simulation"):
            states_history = simulate_markov(transition_matrix, initial_state, num_steps)
            state_labels = [f"État {i}" for i in range(num_states)]
            plot_simulation_markov(states_history, state_labels)
    else:
        st.error("La matrice de transition doit être valide : toutes les probabilités doivent être positives et la somme de chaque ligne doit être égale à 1.")

# Fonction de calcul de la probabilité
def calculate_probability(lam, t, k):
    mean_rate = lam * t  # Calcul du taux moyen d'événements (λt)
    probability = poisson.pmf(k, mean_rate)  # Calcul de la probabilité pour k événements
    return probability

# Fonction de tracé de la CDF (Fonction de répartition cumulative)
def plot_cdf(lam, t):
    mean_rate = lam * t  # Calcul du taux moyen d'événements (λt)
    k_values = np.arange(0, int(mean_rate) + 15)  # Plage de valeurs de k (événements)
    cdf_values = poisson.cdf(k_values, mean_rate)  # Fonction de répartition cumulative
    
    fig, ax = plt.subplots()  # Créer un objet figure et axe explicitement
    ax.plot(k_values, cdf_values, marker='o', color="royalblue", linestyle='-', markersize=7)
    ax.set_title("Fonction de répartition cumulative (CDF)", fontsize=18, color="darkblue")
    ax.set_xlabel("Nombre d'événements (k)", fontsize=14, color="darkslategray")
    ax.set_ylabel("Probabilité cumulée", fontsize=14, color="darkslategray")
    ax.grid(True)
    
    st.pyplot(fig)  # Passer le figure à st.pyplot

# Fonction de tracé du graphique de distribution de Poisson
def plot_graph(lam, t):
    mean_rate = lam * t  # Calcul du taux moyen d'événements (λt)
    k_values = np.arange(0, int(mean_rate) + 15)  # Plage de valeurs de k (événements)
    pmf_values = poisson.pmf(k_values, mean_rate)  # Calcul de la fonction de masse de probabilité
    
    fig, ax = plt.subplots()  # Créer un objet figure et axe explicitement
    ax.bar(k_values, pmf_values, color="lightblue", edgecolor="black", width=0.8)
    ax.set_title("Distribution de Poisson", fontsize=18, color="darkblue")
    ax.set_xlabel("Nombre d'événements (k)", fontsize=14, color="darkslategray")
    ax.set_ylabel("Probabilité", fontsize=14, color="darkslategray")
    ax.grid(True)
    
    st.pyplot(fig)  # Passer le figure à st.pyplot


st.sidebar.title('Processus')

# Grand titre avec un selectbox dans la sidebar pour le choix de la section
grand_titre_1 = st.sidebar.selectbox("Sélectionnez un Processus", 
                                     ["Processus de Poisson Homogène", "Processus de Poisson Non Homogène","processus de Markov à temps discret"])

# Initialisation des variables de sous-section
section_choix_1 = None
section_choix_2 = None
section_choix_3 = None

# Sous-titres en fonction du grand titre sélectionné
if grand_titre_1 == "Processus de Poisson Homogène":
    with st.sidebar.expander("Choisissez une sous-section pour le processus homogène"):
        section_choix_1 = st.selectbox("Choisissez une sous-section", 
                                       ["Simulation", "Ressources"])

elif grand_titre_1 == "Processus de Poisson Non Homogène":
    with st.sidebar.expander("Choisissez une sous-section pour le processus non homogène"):
        section_choix_2 = st.selectbox("Choisissez une sous-section", 
                                       ["Résultats Mathématiques", "Graphique de Simulation", "Ressources"])
elif grand_titre_1 == "processus de Markov à temps discret":
    with st.sidebar.expander("Choisissez une sous-section pour le processus de Markov à temps discret"):
        section_choix_3 = st.selectbox("Choisissez une sous-section", 
                                       ["Simulation", "Ressources"])
# Affichage des résultats en fonction du choix
if grand_titre_1 == "Processus de Poisson Homogène":
    if section_choix_1 == "Simulation":
        # Titre de la page de simulation
        st.title(" Processus de Poisson Homogène")
        st.markdown(
            "<div style='text-align: justify;'>Un processus de Poisson homogène modélise des événements à un taux constant (λ)."
            " La probabilité de k événements dans un intervalle de temps t est donnée par :</div>",
            unsafe_allow_html=True
        )

        # Affichage de la formule en LaTeX
        st.latex(r'''
            P(k; \lambda t) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}
        ''')

        # Explication des paramètres
        st.subheader("Explication des Paramètres")
        st.markdown(""" 
        - **λ (taux moyen d'événements)** : C'est le taux moyen d'occurrence d'événements dans un intervalle de temps.
        - **t (durée de l'intervalle)** : C'est la durée sur laquelle nous mesurons le processus.
        - **k (nombre d'événements)** : C'est le nombre exact d'événements pour lequel nous voulons calculer la probabilité.
        - **P(k; λt)** : C'est la probabilité qu'il y ait exactement k événements pendant l'intervalle de temps t.
        """)

        st.markdown("""
            Cette formule donne la probabilité d'observer exactement k événements dans l'intervalle t, 
            avec un taux moyen de λ événements par unité de temps.
        """)

        # Entrées de l'utilisateur avec des espaces de séparation
        st.markdown("### Paramètres de Simulation")
        lam = st.number_input("Taux moyen d'événements (λ)", min_value=0.0, step=0.1, format="%.2f", help="Entrez un taux moyen d'événements.")
        t = st.number_input("Durée de l'intervalle (t)", min_value=0.0, step=0.1, format="%.2f", help="Entrez la durée de l'intervalle de temps.")
        k = st.number_input("Nombre d'événements (k)", min_value=0, step=1, help="Entrez le nombre exact d'événements.")

        # Calcul de la probabilité
        if st.button("Calculer la probabilité", key="calculate_button"):
            try:
                probability = calculate_probability(lam, t, k)
                st.success(f"**P(k={k}; λt={lam * t:.2f}) = {probability:.4f}**")
            except ValueError:
                st.error("Veuillez entrer des valeurs valides.")

        # Tracé de la CDF
        if st.button("Tracer la fonction de répartition cumulative (CDF)", key="cdf_button"):
            try:
                plot_cdf(lam, t)
            except ValueError:
                st.error("Veuillez entrer des valeurs valides.")

        # Tracé du graphique de distribution de Poisson
        if st.button("Afficher le graphe", key="graph_button"):
            try:
                plot_graph(lam, t)
            except ValueError:
                st.error("Veuillez entrer des valeurs valides.")


    elif section_choix_1 == "Ressources":
        
        st.title("Processus de Poisson Homogène(Ressources)")
        st.markdown(
            "<div style='text-align: justify;'>Voici quelques ressources utiles pour comprendre le processus de Poisson homogène :</div>",
            unsafe_allow_html=True
        )

        # Liens vers les ressources avec icônes
        st.markdown("[📖 Formule du Processus de Poisson Homogène](https://www.wikipedia.org/wiki/Poisson_distribution)")
        st.markdown("[📊 Graphiques et Visualisation](https://www.mathworks.com/help/stats/poisson-distribution.html)")
        st.markdown("[🔍 Explication détaillée du processus de Poisson](https://fr.wikipedia.org/wiki/Processus_de_Poisson)")






elif grand_titre_1 == "Processus de Poisson Non Homogène":
    if section_choix_2 == "Résultats Mathématiques":
        st.title("Processus de Poisson Non Homogène")
        st.markdown(
            "<div style='text-align: justify;'>Un processus de Poisson non homogène modélise des événements dont le taux (λ) varie avec le temps."
            " Le taux d'événements λ(t) dépend d'une fonction du temps. La probabilité de k événements dans un intervalle de temps t est donnée par :</div>",
            unsafe_allow_html=True
        )

        # Affichage de la formule en LaTeX
        st.latex(r'''
            P(k; \int_{t_0}^{t} \lambda(t) dt) = \frac{\left( \int_{t_0}^{t} \lambda(t) dt \right)^k e^{-\int_{t_0}^{t} \lambda(t) dt}}{k!}
        ''')

        # Explication des paramètres
        st.subheader("Explication des Paramètres")
        st.markdown(""" 
        - **λ(t) (taux d'événements)** : C'est la fonction de taux d'occurrence d'événements dans un intervalle de temps. Elle peut être définie par l'utilisateur.
        - **t_start (temps de début)** : C'est l'heure de début de l'intervalle de temps.
        - **t_end (temps de fin)** : C'est l'heure de fin de l'intervalle de temps.
        - **k (nombre d'événements)** : C'est le nombre exact d'événements pour lequel nous voulons calculer la probabilité.
        """)

        # Entrée de la fonction λ(t) définie par l'utilisateur
        lambda_t_input = st.text_input("Définissez la fonction λ(t) comme une fonction Python", value="2 + 0.1 * t", help="Exemple : '2 + 0.1 * t' pour λ(t) = 2 + 0.1*t")
        
        try:
            # Convertir la chaîne en une fonction Python
            lambda_t_func = lambda t: eval(lambda_t_input)
        except Exception as e:
            st.error(f"Erreur dans la fonction λ(t) : {str(e)}")
            lambda_t_func = None

        if lambda_t_func:
            # Entrées de l'utilisateur pour l'intervalle de temps et le nombre d'événements
            t_start = st.number_input("Temps de début (t_start)", min_value=0.0, step=0.1, format="%.2f", help="Entrez l'heure de début de l'intervalle de temps.")
            t_end = st.number_input("Temps de fin (t_end)", min_value=0.0, step=0.1, format="%.2f", help="Entrez l'heure de fin de l'intervalle de temps.")
            k = st.number_input("Nombre d'événements (k)", min_value=0, step=1, help="Entrez le nombre exact d'événements.")
            
            # Calcul de la probabilité
            if st.button("Calculer la probabilité", key="calculate_button"):
                try:
                    probability = calculate_probability_N(lambda_t_func, t_start, t_end, k)
                    st.success(f"**P(k={k}; λ(t)={lambda_t_input} intégrée) = {probability:.6f}**")
                except ValueError:
                    st.error("Veuillez entrer des valeurs valides.")


    elif section_choix_2 == "Graphique de Simulation":
        st.title("Processus de Poisson Non Homogène(Graphique de Simulation)")
        st.markdown(
            "<div style='text-align: justify;'>Cette section permet de visualiser la simulation du processus de Poisson non homogène en traçant le taux d'événements λ(t) en fonction du temps.</div>",
            unsafe_allow_html=True
        )

        # Entrée de la fonction λ(t) définie par l'utilisateur
        lambda_t_input = st.text_input("Définissez la fonction λ(t) comme une fonction Python", value="2 + 0.1 * t", help="Exemple : '2 + 0.1 * t' pour λ(t) = 2 + 0.1*t")
        
        try:
            # Convertir la chaîne en une fonction Python
            lambda_t_func = lambda t: eval(lambda_t_input)
        except Exception as e:
            st.error(f"Erreur dans la fonction λ(t) : {str(e)}")
            lambda_t_func = None

        if lambda_t_func:
            # Entrées de l'utilisateur pour l'intervalle de temps
            t_start = st.number_input("Temps de début (t_start)", min_value=0.0, step=0.1, format="%.2f", help="Entrez l'heure de début de l'intervalle de temps.")
            t_end = st.number_input("Temps de fin (t_end)", min_value=0.0, step=0.1, format="%.2f", help="Entrez l'heure de fin de l'intervalle de temps.")
            
            # Tracé du graphique de simulation
            if st.button("Afficher le graphique", key="graph_button"):
                try:
                    plot_simulation(lambda_t_func, t_start, t_end)
                except ValueError:
                    st.error("Veuillez entrer des valeurs valides.")

    elif section_choix_2 == "Ressources":
        st.title(" Poisson Non Homogène(Ressources)")
        st.title("Ressources sur le Processus de Poisson Non Homogène")
        st.markdown(
            "<div style='text-align: justify;'>Voici quelques ressources utiles pour comprendre le processus de Poisson non homogène :</div>",
            unsafe_allow_html=True
        )

        # Liens vers les ressources avec icônes
        st.markdown("[📖 Formule du Processus de Poisson Non Homogène](https://en.wikipedia.org/wiki/Poisson_point_process#Inhomogeneous_Poisson_point_process)")
        st.markdown("[📊 Graphiques et Visualisation(exemple1)](https://en.wikipedia.org/wiki/Poisson_point_process#/media/File:Inhomogeneouspoissonprocess.svg)")
        st.markdown("[📊 Graphiques et Visualisation(exemple2)](https://x-datainitiative.github.io/tick/auto_examples/plot_poisson_inhomogeneous.html)")
        st.markdown("[🔍 Explication détaillée du processus de Poisson](https://gtribello.github.io/mathNET/resources/jim-chap22.pdf)")

elif grand_titre_1 == "processus de Markov à temps discret":
    if section_choix_3== "Simulation":
        st.subheader("processus de Markov à temps discret")
            
        st.markdown(""" 
            Le **processus de Markov à temps discret** est un modèle où l'état futur dépend uniquement de l'état actuel. Il est décrit par une matrice de transition \( P \), où chaque élément \( P_{ij} \) représente la probabilité de transition de l'état \( i \) à l'état \( j \). La relation est exprimée par l'équation suivante :
        """)

        st.latex(r'''
            P_{ij} = P\left( \text{État } j \text{ au temps } t+1 \mid \text{État } i \text{ au temps } t \right)
        ''')

        st.markdown(""" 
            De plus, la somme des probabilités de transition depuis chaque état \( i \) est égale à 1 :
        """)

        st.latex(r'''
            \sum_j P_{ij} = 1
        ''')

        # Définir le nombre d'états
        num_states = st.number_input("Nombre d'états", min_value=2, step=1, value=3)

        # Choisir la méthode pour définir la matrice de transition
        matrix_option = st.radio(
            "Choisissez comment définir la matrice de transition",
            ["Générer aléatoirement", "Saisir manuellement"]
        )
        
        # Initialiser la matrice de transition
        transition_matrix = []

        if matrix_option == "Générer aléatoirement":
            # Choisir la distribution pour générer la matrice de transition
            distribution_type = st.selectbox(
                "Choisissez la distribution pour générer la matrice de transition",
                ["Distribution uniforme", "Distribution de Dirichlet", "Distribution normale"]
            )
            
            # Afficher la définition de la distribution choisie
            if distribution_type == "Distribution uniforme":
                st.markdown(r"La **distribution uniforme** génère des probabilités de transition égales entre tous les états, avec $P_{ij} = \frac{1}{n}$, assurant que chaque état a une probabilité égale d'être suivi d'un autre.")

            elif distribution_type == "Distribution de Dirichlet":
                st.markdown(r"La **distribution de Dirichlet** génère des matrices de transition où chaque ligne somme à 1, définie par $P_{ij} \sim \text{Dirichlet}(\alpha_1, \alpha_2, \dots, \alpha_n)$, avec $\alpha_i$ comme paramètre positif déterminant la concentration des probabilités.")

            elif distribution_type == "Distribution normale":
                st.markdown(r"La **distribution normale**  génère des probabilités suivant une gaussienne centrée autour de  \( 0.5 \)  avec un écart-type de  \( 0.1 \)  , normalisées pour garantir que la somme de chaque ligne soit égale à 1, soit    ,"
                            r"$ P_{ij} \sim \mathcal{N}(0.5, 0.1) $")



            
            # Générer la matrice de transition en fonction de la distribution choisie
            transition_matrix = generate_transition_matrix(num_states, distribution_type)
            st.subheader("Matrice de Transition Générée")
            st.write(transition_matrix)
            
    
        elif matrix_option == "Saisir manuellement":
            st.subheader("Matrice de Transition Manuelle")
            for i in range(num_states):
                row = st.text_input(f"Probabilités de transition depuis l'état {i}", value="0.5, 0.5", help="Exemple : '0.5, 0.5' pour une matrice 2x2.")
                try:
                    transition_matrix.append(list(map(float, row.split(','))))
                except ValueError:
                    st.warning(f"Les valeurs de la ligne {i} doivent être des nombres séparés par des virgules.")
            
            transition_matrix = np.array(transition_matrix)

            # Vérification si chaque ligne contient bien 'num_states' éléments
            if transition_matrix.shape[1] != num_states:
                st.warning(f"Chaque ligne de la matrice doit avoir {num_states} éléments.")
                transition_matrix = []

        # Vérifier si la matrice de transition est valide
        if is_valid_transition_matrix(transition_matrix):
            # Définir l'état initial
            initial_state = st.selectbox("Choisissez l'état initial", range(num_states))
            
            # Nombre d'étapes de la simulation
            num_steps = st.number_input("Nombre d'étapes de simulation", min_value=1, step=1, value=10)
            
            # Étiquettes des états
            state_labels = [f"État {i}" for i in range(num_states)]
            
            # Simuler le processus de Markov
            if st.button("Lancer la simulation", key="simulate_button"):
                states_history = simulate_markov(transition_matrix, initial_state, num_steps)
                plot_simulation_markov(states_history, state_labels)
        else:
            st.error("La matrice de transition doit être valide : toutes les probabilités doivent être positives et la somme de chaque ligne doit être égale à 1.")
    elif section_choix_3 == "Ressources":
        # Titre de la page de ressources
        st.title("Ressources sur le processus de Markov à temps discret")
        
        # Liens vers les ressources avec icônes
        st.markdown("\n\n")
        st.markdown("[📖 Introduction aux Processus de Markov](https://fr.wikipedia.org/wiki/Cha%C3%AEne_de_Markov)")
        st.markdown("[📊 Exemple d'un processus de Markov simple](https://www.mathworks.com/help/stats/markov-chains.html)")
        st.markdown("[📚 Distribution Normal](https://fr.wikipedia.org/wiki/Loi_normale.)")
        st.markdown("[📘 Distribution uniform](https://www.probabilitycourse.com/chapter4/4_2_1_uniform.php)")
        st.markdown("[📊 Distribution Dirichlet](https://fr.wikipedia.org/wiki/Loi_de_Dirichlet)")




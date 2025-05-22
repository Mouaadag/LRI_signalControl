# SignalControl - Système de Recommandation par Renforcement

## Présentation du Projet

SignalControl est un projet de recherche appliquée en apprentissage par renforcement (Reinforcement Learning), dédié à l’optimisation dynamique des systèmes de contrôle. Son objectif est de permettre l’ajustement intelligent d’un signal de commande en temps réel, de manière à concilier efficacité énergétique et confort utilisateur, souvent en tension dans les systèmes réels tels que le chauffage, la climatisation ou l’éclairage.

Le cœur du projet repose sur un modèle d’apprentissage par renforcement capable de s’adapter à des environnements variés pour atteindre un équilibre optimal entre consommation d’énergie et satisfaction des utilisateurs. Ce modèle est exposé via une API REST, facilitant son intégration dans une application web interactive.

Cette application permet de visualiser et d’interagir avec le comportement de l’agent en temps réel, dans des scénarios simulés. Elle offre ainsi une démonstration concrète de la capacité des algorithmes de RL à s’ajuster dynamiquement, tout en rendant accessible une technologie avancée à travers une interface intuitive et pédagogique.

## Fonctionnalités principales

- **Agent d'apprentissage par renforcement** : Modèle adaptatif qui apprend des interactions passées pour optimiser ses recommandations
- **Interface utilisateur interactive** : Tableau de bord complet avec visualisations en temps réel
- **Modes de fonctionnement multiples** :
  - Mode manuel avec recommandations
  - Mode automatique pour des ajustements sans intervention
  - Mode d'analyse historique pour générer un feedback pertinent
- **Métriques de performance** : Suivi des économies d'énergie et du confort utilisateur
- **Visualisation graphique** : Affichage de l'évolution du signal dans le temps
- **Historique des prédictions** : Journal détaillé des actions et recommandations

## Installation

### Prérequis
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (gestionnaire de packages Python )

### Instructions

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/SignalControl.git
   cd SignalControl
   ```

2. Créez un environnement virtuel avec uv :
   ```bash
   uv venv
   ```

3. Activez l'environnement virtuel :
   ```bash
   source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
   ```

4. Installez les dépendances avec uv :
   ```bash
   uv pip install -r requirements.txt
   ```

5. Lancez l'application :
   ```bash
   python app.py
   ```

6. Ouvrez votre navigateur à l'adresse : `http://localhost:5000`

## Guide d'utilisation

### Paramètres du système

Dans l'interface, vous pouvez configurer :
- **Valeur maximale du signal** : Valeur plafond du système
- **Taille du pas** : Granularité des ajustements
- **Intervalle de mise à jour** : Fréquence des ajustements automatiques

### Modes d'interaction

1. **Mode recommandation** : L'agent suggère des modifications de valeur
2. **Mode automatique** : Le système applique automatiquement les recommandations à intervalle régulier
3. **Mode intervention** : Simulation d'une intervention utilisateur (remet le signal à sa valeur maximale)
4. **Mode feedback historique** : Analyse les données historiques pour générer un feedback pertinent

### Métriques de performance

- **Économie d'énergie** : Pourcentage d'économie par rapport à la valeur maximale
- **Score de confort** : Évaluation du confort utilisateur basée sur la fréquence des interventions
- **Nombre d'interventions** : Total des interventions manuelles effectuées

## Fonctionnement technique

### Algorithme d'apprentissage

L'application utilise un modèle d'apprentissage par renforcement qui optimise un triplet de probabilités :
- Probabilité de **diminuer** le signal
- Probabilité de **maintenir** le signal
- Probabilité d'**augmenter** le signal

Ces probabilités sont ajustées en fonction des récompenses obtenues suite aux actions précédentes.

### Système de récompenses

- **Récompense positive** : Pour les économies d'énergie (signal plus bas)
- **Récompense négative** : En cas d'intervention utilisateur (signal trop bas)

## Architecture du projet

- **app.py** : Application Flask principale et API
- **model/** : Contient le modèle d'apprentissage par renforcement
- **notebook** : Contient le jupyter notebook qui contient les expérience de convergence de notre model
-- **article** : Contient l'article scientifique de cet approche
- **templates/** : Fichiers HTML pour l'interface utilisateur
- **static/** : Ressources statiques (CSS, JS)

## Dépendances principales

- Flask : Framework web
- NumPy : Calculs mathématiques
- Plotly : Visualisations interactives
- Pandas : Manipulation des données

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Soumettre une pull request

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
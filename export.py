"""
Script d'exportation du modèle LRI pour SignalControl

Ce script permet d'exporter un modèle d'apprentissage par renforcement LRI (Linear Reward-Inaction)
préalablement optimisé vers un fichier pickle pour une utilisation en production.
Il prend en paramètre un triplet de probabilités optimal (diminuer/maintenir/augmenter)
ainsi que des hyperparamètres comme la taille du pas et le facteur gamma qui équilibre
économie d'énergie et confort utilisateur.

Usage:
    python export.py  # Exporte avec les valeurs par défaut
    
Le modèle exporté sera utilisé par l'application web SignalControl pour générer
des recommandations d'ajustement du signal de contrôle.
"""

import os
import pickle
import sys

import numpy as np

# Ajouter le chemin du notebook/ProduitFinal au système
sys.path.append(os.path.abspath("../notebook/"))

# Importer les classes nécessaires
from LRIagent import LRIAgent
from UserModel import UserModel


def export_model(best_triplet, step_size=5, gamma=0.35, filepath="model/lri_model.pkl"):
    """
    Exporte un modèle LRI optimisé pour la production.
    
    Paramètres:
    -----------
    best_triplet : array-like
        Triplet de probabilités optimal identifié lors des expériences
    step_size : float
        Taille du pas d'augmentation/diminution du signal
    gamma : float
        Facteur de pondération entre économie d'énergie et confort
    filepath : str
        Chemin où sauvegarder le modèle
    """
    # Créer un agent avec le triplet optimal
    lri_agent = LRIAgent(triplets=[best_triplet], step_size=step_size, gamma=gamma)
    
    # Créer le dossier de destination si nécessaire
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Sauvegarder le modèle
    with open(filepath, 'wb') as f:
        pickle.dump(lri_agent, f)
    
    print(f"Modèle LRI exporté avec succès vers {filepath}")
    return lri_agent

if __name__ == "__main__":
    # Utilisez le meilleur triplet de l'expérience 3 (à remplacer par vos valeurs optimales)
    best_triplet = [0.088, 0.876, 0.036]  # [p_dim, p_maint, p_aug]
    
    # Paramètres optimaux identifiés
    step_size = 5
    gamma = 0.45
    
    # Exporter le modèle
    export_model(best_triplet, step_size, gamma)
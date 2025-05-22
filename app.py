import datetime
import json
import os
import pickle
import sys
import time
from io import BytesIO

import numpy as np

# Utilisation de Plotly pour les graphiques
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, jsonify, render_template, request, send_file

# Ajouter les chemins nécessaires
sys.path.append(os.path.abspath("../notebook/ProduitFinal"))

app = Flask(__name__)

# Charger le modèle
MODEL_PATH = "model/lri_model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

# État actuel du système - stocké en mémoire (dans un cas réel, utilisez une base de données)
system_state = {
    "current_value": 90,
    "last_update": datetime.datetime.now().isoformat(),
    "history": [],
    "signal_history": [],  # Historique complet des valeurs pour le graphique
    "timestamp_history": [],  # Horodatages correspondants
    "max_signal": 90,  # Valeur maximale du signal
    "step_size": 5,   # Taille du pas
    "update_interval": 60,  # Intervalle de mise à jour en secondes
    "total_interventions": 0,  # Nombre total d'interventions utilisateur
    "energy_metrics": {
        "baseline_consumption": 0,       # Consommation sans modèle (référence)
        "current_consumption": 0,        # Consommation avec modèle
        "energy_savings_percent": 0,     # Économies en pourcentage
        "total_interventions": 0,        # Nombre total d'interventions utilisateur
        "comfort_score": 100,            # Score de confort (100 = parfait)
        "last_calculation": None         # Dernière mise à jour des métriques
    }
}

# Charger l'état précédent si disponible
STATE_PATH = "model/system_state.json"
try:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r') as f:
            saved_state = json.load(f)
            system_state.update(saved_state)
            print(f"État du système chargé depuis {STATE_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement de l'état du système: {e}")

# Sauvegarder l'état périodiquement
def save_system_state():
    try:
        # Créer une copie pour la sauvegarde (éviter les erreurs de sérialisation)
        state_to_save = system_state.copy()
        
        # Limiter la taille des historiques pour la sauvegarde
        if len(state_to_save["signal_history"]) > 1000:
            # Garder seulement les 1000 dernières valeurs
            state_to_save["signal_history"] = state_to_save["signal_history"][-1000:]
            state_to_save["timestamp_history"] = state_to_save["timestamp_history"][-1000:]
        
        with open(STATE_PATH, 'w') as f:
            json.dump(state_to_save, f)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'état du système: {e}")

# Calculer les métriques d'économie d'énergie et de confort
def calculate_energy_metrics():
    """Calcule les métriques d'économie d'énergie et de confort"""
    try:
        signal_history = system_state["signal_history"]
        if not signal_history or len(signal_history) < 2:
            return  # Pas assez de données pour calculer
        
        # Calculer la consommation de référence (sans modèle)
        # Hypothèse: la valeur maximale est la consommation par défaut
        max_signal = system_state["max_signal"]
        baseline_consumption = max_signal * len(signal_history)
        
        # Calculer la consommation actuelle avec le modèle
        current_consumption = sum(signal_history)
        
        # Calculer le pourcentage d'économie d'énergie
        if baseline_consumption > 0:
            energy_savings_percent = ((baseline_consumption - current_consumption) / baseline_consumption) * 100
        else:
            energy_savings_percent = 0
            
        # Récupérer le nombre d'interventions utilisateur
        interventions = system_state.get("total_interventions", 0)
        
        # Calculer le score de confort - Version plus sensible
        # Si le nombre d'interventions dépasse 20% du nombre de points, le score commence à baisser
        if len(signal_history) > 0:
            # Rendre la formule plus sensible aux interventions
            comfort_ratio = min(1.0, interventions / (len(signal_history) * 0.2))
            comfort_score = 100 * (1 - comfort_ratio)
        else:
            comfort_score = 100
        
        # Mettre à jour les métriques
        system_state["energy_metrics"].update({
            "baseline_consumption": baseline_consumption,
            "current_consumption": current_consumption,
            "energy_savings_percent": round(energy_savings_percent, 2),
            "total_interventions": interventions,
            "comfort_score": round(comfort_score, 2),
            "last_calculation": datetime.datetime.now().isoformat()
        })
        
        print(f"Métriques mises à jour: Économie d'énergie: {energy_savings_percent:.2f}%, "
              f"Score de confort: {comfort_score:.2f}, Interventions: {interventions}")
    
    except Exception as e:
        print(f"Erreur lors du calcul des métriques: {e}")

# Historique des prédictions pour démonstration
prediction_history = system_state.get("history", [])

@app.route('/')
def index():
    """Page d'accueil avec interface utilisateur"""
    # Ne pas incrémenter le compteur d'interventions lors d'un simple chargement de page
    return render_template('index.html', 
                          history=prediction_history,
                          current_value=system_state["current_value"],
                          max_signal=system_state["max_signal"],
                          step_size=system_state["step_size"],
                          update_interval=system_state["update_interval"])

@app.route('/settings', methods=['POST'])
def update_settings():
    """Mise à jour des paramètres du système"""
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
            
        # Mettre à jour les paramètres si fournis
        if 'max_signal' in data:
            system_state["max_signal"] = float(data['max_signal'])
        
        if 'step_size' in data:
            system_state["step_size"] = float(data['step_size'])
            # Mettre à jour le modèle avec la nouvelle taille de pas
            if model:
                model.step_size = system_state["step_size"]
        
        if 'update_interval' in data:
            system_state["update_interval"] = int(data['update_interval'])
        
        # Incrémenter le compteur d'interventions
        system_state["total_interventions"] = system_state.get("total_interventions", 0) + 1
        
        # MODIFICATION: Remettre le signal à sa valeur maximale lors d'une intervention
        system_state["current_value"] = system_state["max_signal"]
        
        # Ajouter cette valeur à l'historique du signal
        system_state["signal_history"].append(system_state["max_signal"])
        system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
        
        # Sauvegarder les modifications
        save_system_state()
        
        # Recalculer les métriques
        calculate_energy_metrics()
        
        if request.is_json:
            return jsonify({
                "success": True, 
                "settings": {
                    "max_signal": system_state["max_signal"],
                    "step_size": system_state["step_size"],
                    "update_interval": system_state["update_interval"]
                },
                "message": "Paramètres mis à jour. Signal remis à sa valeur maximale."
            })
        else:
            return render_template('index.html', 
                                  success_message="Paramètres mis à jour avec succès. Signal remis à sa valeur maximale.",
                                  history=prediction_history,
                                  current_value=system_state["current_value"],
                                  max_signal=system_state["max_signal"],
                                  step_size=system_state["step_size"],
                                  update_interval=system_state["update_interval"])
                                  
    except Exception as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        else:
            return render_template('index.html', 
                                  error=error_msg,
                                  history=prediction_history,
                                  current_value=system_state["current_value"],
                                  max_signal=system_state["max_signal"],
                                  step_size=system_state["step_size"],
                                  update_interval=system_state["update_interval"])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Endpoint pour obtenir une prédiction automatiquement"""
    if model is None:
        return jsonify({"error": "Modèle non disponible"}), 503
    
    try:
        # Valeur actuelle provient de l'état du système
        current_value = system_state["current_value"]
        
        # Si l'utilisateur a fourni une valeur explicite, utilisez-la (prioritaire)
        if request.method == 'POST':
            if request.is_json:
                data = request.get_json()
                if "current_value" in data:
                    current_value = float(data.get('current_value'))
            else:
                form_value = request.form.get('current_value')
                if form_value:
                    current_value = float(form_value)
                
            # Incrémenter le compteur d'interventions pour les requêtes POST
            system_state["total_interventions"] = system_state.get("total_interventions", 0) + 1
            
            # MODIFICATION: En cas d'intervention, remettre à la valeur max
            current_value = system_state["max_signal"]
            # Ajouter cette valeur à l'historique
            system_state["signal_history"].append(current_value)
            system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
        
        # Mettre à jour la step_size du modèle avec celle du système
        model.step_size = system_state["step_size"]
        
        # Obtenir la nouvelle valeur recommandée
        new_value = model.select_action(current_value)
        
        # S'assurer que la valeur ne dépasse pas max_signal
        new_value = min(new_value, system_state["max_signal"])
        
        # Obtenir le triplet actuel
        current_triplet = model.get_probabilities().tolist() 
        
        # Enregistrer dans l'historique
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        prediction_entry = {
            'timestamp': timestamp,
            'current_value': current_value,
            'new_value': new_value,
            'triplet': current_triplet
        }
        prediction_history.insert(0, prediction_entry)
        
        # Limiter l'historique à 10 entrées
        if len(prediction_history) > 10:
            prediction_history.pop()
        
        # Ajouter à l'historique pour le graphique
        system_state["signal_history"].append(new_value)
        system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
        
        # Mettre à jour l'état du système
        system_state["current_value"] = new_value
        system_state["last_update"] = datetime.datetime.now().isoformat()
        system_state["history"] = prediction_history
        
        # Mettre à jour les métriques
        calculate_energy_metrics()
        
        # Sauvegarder l'état
        save_system_state()
        
        # Retourner la prédiction
        result = {
            'timestamp': timestamp,
            'current_value': current_value,
            'recommended_value': new_value,
            'current_triplet': current_triplet,
            'intervention': request.method == 'POST'  # Indiquer s'il s'agit d'une intervention
        }
        
        if request.method == 'GET' or (request.method == 'POST' and not request.is_json):
            return render_template('index.html', 
                                  result=result, 
                                  history=prediction_history,
                                  current_value=system_state["current_value"],
                                  max_signal=system_state["max_signal"],
                                  step_size=system_state["step_size"],
                                  update_interval=system_state["update_interval"])
        else:
            return jsonify(result)
    
    except Exception as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        else:
            return render_template('index.html', 
                                  error=error_msg,
                                  history=prediction_history,
                                  current_value=system_state["current_value"],
                                  max_signal=system_state["max_signal"],
                                  step_size=system_state["step_size"],
                                  update_interval=system_state["update_interval"])

@app.route('/auto_feedback', methods=['POST'])
def auto_feedback():
    """Endpoint pour générer et envoyer un feedback automatique basé sur l'historique"""
    if model is None:
        return jsonify({"error": "Modèle non disponible"}), 503
    
    try:
        data = request.get_json()
        history_length = data.get('history_length', '10')
        feedback_mode = data.get('feedback_mode', 'avg')
        no_intervention = data.get('no_intervention', False)  # Nouveau paramètre
        
        # Récupérer l'historique du signal
        signal_history = system_state["signal_history"]
        
        # Déterminer combien de points utiliser
        if history_length == 'all':
            points_to_use = len(signal_history)
        else:
            points_to_use = min(int(history_length), len(signal_history))
        
        if points_to_use < 2:
            return jsonify({"error": "Pas assez de données dans l'historique pour générer un feedback"}), 400
        
        # Obtenir la partie pertinente de l'historique
        relevant_history = signal_history[-points_to_use:]
        
        # Calculer la valeur optimale selon le mode choisi
        if feedback_mode == 'avg':
            # Utiliser la moyenne des valeurs
            optimal_value = sum(relevant_history) / len(relevant_history)
        elif feedback_mode == 'min':
            # Utiliser la valeur minimale qui a bien fonctionné
            # Pour éviter les valeurs extrêmes, prendre le 10e percentile
            optimal_value = sorted(relevant_history)[max(0, int(len(relevant_history) * 0.1))]
        elif feedback_mode == 'comfort':
            # Privilégier le confort - prendre une valeur qui a nécessité moins d'interventions
            # Pour simplifier, utiliser une valeur entre la moyenne et le maximum
            avg_value = sum(relevant_history) / len(relevant_history)
            max_value = max(relevant_history)
            optimal_value = (avg_value + max_value) / 2
        else:
            optimal_value = system_state["current_value"]
        
        # Calculer une durée et un temps d'inactivité virtuels basés sur l'historique
        virtual_duration = points_to_use * system_state["update_interval"]
        virtual_idle_time = virtual_duration * 0.8  # 80% du temps total comme estimation
        
        # Calculer la récompense
        reward = model.calculate_reward(optimal_value, virtual_duration, virtual_idle_time)
        
        # Utiliser votre formule pour mettre à jour le modèle
        # La méthode update() de votre modèle doit utiliser:
        # self.triplet_probs = self.triplet_probs + self.alpha * reward * (e_i - self.triplet_probs)
        model.update(reward)
        
        # Extraire le nouveau triplet
        new_triplet = model.get_best_triplet().tolist()
        
        # Gérer l'intervention en fonction du paramètre no_intervention
        if not no_intervention:
            # Incrémenter le compteur d'interventions
            system_state["total_interventions"] = system_state.get("total_interventions", 0) + 1
            
            # Remettre le signal à sa valeur maximale
            system_state["current_value"] = system_state["max_signal"]
            
            # Ajouter cette valeur à l'historique du signal
            system_state["signal_history"].append(system_state["max_signal"])
            system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
            
            # Mettre à jour les métriques
            calculate_energy_metrics()
        
        # Sauvegarder le modèle mis à jour
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        # Sauvegarder l'état du système
        save_system_state()
        
        return jsonify({
            "success": True,
            "points_analyzed": points_to_use,
            "optimal_value": float(optimal_value),
            "reward": float(reward),
            "new_triplet": new_triplet,
            "feedback_mode": feedback_mode,
            "current_value": system_state["current_value"],
            "intervention": not no_intervention  # Indiquer si une intervention a eu lieu
        })
    
    except Exception as e:
        print(f"Erreur lors du feedback automatique: {e}")
        return jsonify({"error": str(e)}), 400
    
@app.route('/current_value', methods=['GET'])
def get_current_value():
    """Endpoint pour obtenir la valeur actuelle du système"""
    return jsonify({
        "current_value": system_state["current_value"],
        "last_update": system_state["last_update"]
    })

@app.route('/current_value', methods=['POST'])
def set_current_value():
    """Endpoint pour définir manuellement la valeur actuelle du système"""
    try:
        data = request.get_json()
        
        # Incrémenter le compteur d'interventions
        system_state["total_interventions"] = system_state.get("total_interventions", 0) + 1
        
        # MODIFICATION: Remettre le signal à sa valeur maximale lors d'une intervention
        system_state["current_value"] = system_state["max_signal"]
        system_state["last_update"] = datetime.datetime.now().isoformat()
        
        # Ajouter à l'historique pour le graphique
        system_state["signal_history"].append(system_state["max_signal"])
        system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
        
        # Mettre à jour les métriques
        calculate_energy_metrics()
        
        # Sauvegarder l'état
        save_system_state()
        
        return jsonify({
            "success": True,
            "current_value": system_state["current_value"],
            "last_update": system_state["last_update"],
            "message": "Intervention détectée, signal remis à sa valeur maximale"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/auto', methods=['GET'])
def auto_recommend():
    """Endpoint pour obtenir automatiquement une recommandation basée sur la valeur actuelle"""
    if model is None:
        return jsonify({"error": "Modèle non disponible"}), 503
    
    try:
        # Valeur actuelle
        current_value = system_state["current_value"]
        
        # Mettre à jour la step_size du modèle avec celle du système
        model.step_size = system_state["step_size"]
        
        # Obtenir la nouvelle valeur recommandée
        new_value = model.select_action(current_value)
        
        # S'assurer que la valeur ne dépasse pas max_signal
        new_value = min(new_value, system_state["max_signal"])
        
        # Mettre à jour l'état du système
        system_state["current_value"] = new_value
        system_state["last_update"] = datetime.datetime.now().isoformat()
        
        # Ajouter à l'historique pour le graphique
        system_state["signal_history"].append(new_value)
        system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
        
        # Mettre à jour les métriques
        calculate_energy_metrics()
        
        # Sauvegarder l'état
        save_system_state()
        
        return jsonify({
            "previous_value": current_value,
            "recommended_value": new_value,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/graph', methods=['GET'])
def get_graph():
    """Génère et renvoie un graphique de l'historique des valeurs en utilisant Plotly"""
    try:
        # Récupérer les données d'historique
        signal_history = system_state["signal_history"]
        
        # Créer une figure Plotly
        fig = go.Figure()
        
        if not signal_history:
            # Si pas de données, créer un graphique vide avec un message
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        else:
            # Créer des x pour l'axe temporel (indices réguliers)
            x = list(range(len(signal_history)))
            
            # Tracer la courbe principale
            fig.add_trace(go.Scatter(
                x=x,
                y=signal_history,
                mode='lines',
                name='Signal',
                line=dict(color='blue', width=2)
            ))
            
            # Si assez de points, ajouter une moyenne mobile
            if len(signal_history) > 5:
                window_size = min(5, len(signal_history) // 5)
                moving_avg = np.convolve(signal_history, np.ones(window_size)/window_size, mode='valid')
                x_avg = list(range(window_size-1, len(signal_history)))
                
                fig.add_trace(go.Scatter(
                    x=x_avg,
                    y=moving_avg,
                    mode='lines',
                    name=f'Moyenne mobile ({window_size} points)',
                    line=dict(color='red', width=1.5, dash='dash')
                ))
            
            # Ajouter la valeur actuelle comme point
            if signal_history:
                fig.add_trace(go.Scatter(
                    x=[len(signal_history)-1],
                    y=[signal_history[-1]],
                    mode='markers',
                    name=f'Valeur actuelle: {signal_history[-1]}',
                    marker=dict(color='green', size=10)
                ))
            
            # Ajouter la ligne de référence (valeur maximale)
            fig.add_trace(go.Scatter(
                x=[0, len(signal_history)-1],
                y=[system_state["max_signal"], system_state["max_signal"]],
                mode='lines',
                name='Valeur sans modèle',
                line=dict(color='darkgrey', width=1.5, dash='dot')
            ))
        
        # Configurer la mise en page
        fig.update_layout(
            title="Évolution du signal dans le temps",
            xaxis_title="Points de mesure",
            yaxis_title="Valeur du signal",
            yaxis=dict(range=[0, system_state["max_signal"] * 1.1]),
            template="plotly_white",
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=80, pad=4),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Ajouter une grille
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # Convertir en image PNG
        img_bytes = pio.to_image(fig, format="png", width=800, height=500)
        
        # Renvoyer l'image
        img_buf = BytesIO(img_bytes)
        img_buf.seek(0)
        
        return send_file(img_buf, mimetype='image/png')
    
    except Exception as e:
        print(f"Erreur lors de la génération du graphique: {e}")
        
        # Créer une image d'erreur simple avec Plotly
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erreur: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color='red')
        )
        
        # Convertir en image PNG
        img_bytes = pio.to_image(fig, format="png", width=800, height=500)
        
        # Renvoyer l'image
        img_buf = BytesIO(img_bytes)
        img_buf.seek(0)
        
        return send_file(img_buf, mimetype='image/png')

@app.route('/graph_html', methods=['GET'])
def get_graph_html():
    """Génère et renvoie un graphique interactif de l'historique des valeurs"""
    try:
        # Récupérer les données d'historique
        signal_history = system_state["signal_history"]
        
        # Créer une figure Plotly
        fig = go.Figure()
        
        if not signal_history:
            # Si pas de données, créer un graphique vide avec un message
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        else:
            # Créer des x pour l'axe temporel (indices réguliers)
            x = list(range(len(signal_history)))
            
            # Tracer la courbe principale
            fig.add_trace(go.Scatter(
                x=x,
                y=signal_history,
                mode='lines',
                name='Signal',
                line=dict(color='blue', width=2)
            ))
            
            # Si assez de points, ajouter une moyenne mobile
            if len(signal_history) > 5:
                window_size = min(5, len(signal_history) // 5)
                moving_avg = np.convolve(signal_history, np.ones(window_size)/window_size, mode='valid')
                x_avg = list(range(window_size-1, len(signal_history)))
                
                fig.add_trace(go.Scatter(
                    x=x_avg,
                    y=moving_avg,
                    mode='lines',
                    name=f'Moyenne mobile ({window_size} points)',
                    line=dict(color='red', width=1.5, dash='dash')
                ))
            
            # Ajouter la valeur actuelle comme point
            if signal_history:
                fig.add_trace(go.Scatter(
                    x=[len(signal_history)-1],
                    y=[signal_history[-1]],
                    mode='markers',
                    name=f'Valeur actuelle: {signal_history[-1]}',
                    marker=dict(color='green', size=10)
                ))
                
            # Ajouter la ligne de référence (valeur maximale)
            fig.add_trace(go.Scatter(
                x=[0, len(signal_history)-1],
                y=[system_state["max_signal"], system_state["max_signal"]],
                mode='lines',
                name='Valeur sans modèle',
                line=dict(color='darkgrey', width=1.5, dash='dot')
            ))
        
        # Configurer la mise en page
        fig.update_layout(
            title="Évolution du signal dans le temps",
            xaxis_title="Points de mesure",
            yaxis_title="Valeur du signal",
            yaxis=dict(range=[0, system_state["max_signal"] * 1.1]),
            template="plotly_white",
            autosize=True,
            margin=dict(l=50, r=50, b=50, t=80, pad=4),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Ajouter une grille
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        # Convertir en HTML
        graph_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        return graph_html
    
    except Exception as e:
        print(f"Erreur lors de la génération du graphique HTML: {e}")
        # Créer un message d'erreur simple
        return f'<div style="color:red; text-align:center; padding:20px;">Erreur: {str(e)}</div>'

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Endpoint pour obtenir les métriques de performance"""
    # Recalculer les métriques à la demande
    calculate_energy_metrics()
    
    return jsonify({
        "energy_metrics": system_state["energy_metrics"],
        "signal_count": len(system_state["signal_history"]),
        "current_value": system_state["current_value"],
        "max_signal": system_state["max_signal"]
    })
# @app.route('/history')
# def get_history():
#     # Assurez-vous que le header Content-Type est correctement défini
#     return jsonify({
#         'history': system_state.get('history', [])
#     }) 
@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "current_value": system_state["current_value"],
        "max_signal": system_state["max_signal"],
        "step_size": system_state["step_size"],
        "update_interval": system_state["update_interval"],
        "history_size": len(system_state["signal_history"]),
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/start_auto_update', methods=['POST'])
def start_auto_update():
    """Démarre ou arrête la mise à jour automatique"""
    try:
        data = request.get_json() if request.is_json else request.form
        auto_update = data.get('auto_update', 'false').lower() == 'true'
        
        # Cette fonction ne gère que le démarrage du thread de mise à jour
        # Le thread lui-même est implémenté côté client en JavaScript
        
        if auto_update:
            return jsonify({
                "success": True,
                "message": "Mise à jour automatique activée",
                "update_interval": system_state["update_interval"]
            })
        else:
            return jsonify({
                "success": True,
                "message": "Mise à jour automatique désactivée"
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    os.makedirs("templates", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    print("====================================")
    print("Démarrage de l'API LRI sur http://localhost:5005")
    print("Dossier de travail:", os.getcwd())
    print("====================================")
    
    # Lancer l'application en mode développement
    app.run(host='localhost', port=5005, debug=True)
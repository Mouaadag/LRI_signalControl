import datetime
import json
import os
import pickle
import sys
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ajouter les chemins nécessaires
sys.path.append(os.path.abspath("../notebook/"))

# Configuration de la page
st.set_page_config(
    page_title="Signal Control LRI",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le modèle
@st.cache_resource
def load_model():
    MODEL_PATH = "model/lri_model.pkl"
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        st.sidebar.success(f"Modèle chargé depuis {MODEL_PATH}")
        return model
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_model()

# État actuel du système - stocké dans la session
if 'system_state' not in st.session_state:
    # État par défaut
    st.session_state.system_state = {
        "current_value": 90,
        "last_update": datetime.datetime.now().isoformat(),
        "history": [],
        "signal_history": [],
        "timestamp_history": [],
        "max_signal": 90,
        "step_size": 5,
        "update_interval": 60,
        "total_interventions": 0,
        "energy_metrics": {
            "baseline_consumption": 0,
            "current_consumption": 0,
            "energy_savings_percent": 0,
            "total_interventions": 0,
            "comfort_score": 100,
            "last_calculation": None
        }
    }
    
    # Charger l'état précédent si disponible
    STATE_PATH = "model/system_state.json"
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r') as f:
                saved_state = json.load(f)
                st.session_state.system_state.update(saved_state)
                st.sidebar.success(f"État du système chargé depuis {STATE_PATH}")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement de l'état du système: {e}")

# Sauvegarder l'état périodiquement
def save_system_state():
    try:
        # Créer une copie pour la sauvegarde (éviter les erreurs de sérialisation)
        state_to_save = st.session_state.system_state.copy()
        
        # Limiter la taille des historiques pour la sauvegarde
        if len(state_to_save["signal_history"]) > 1000:
            # Garder seulement les 1000 dernières valeurs
            state_to_save["signal_history"] = state_to_save["signal_history"][-1000:]
            state_to_save["timestamp_history"] = state_to_save["timestamp_history"][-1000:]
        
        STATE_PATH = "model/system_state.json"
        with open(STATE_PATH, 'w') as f:
            json.dump(state_to_save, f)
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde de l'état du système: {e}")

# Calculer les métriques d'économie d'énergie et de confort
def calculate_energy_metrics():
    try:
        signal_history = st.session_state.system_state["signal_history"]
        if not signal_history or len(signal_history) < 2:
            return  # Pas assez de données pour calculer
        
        # Calculer la consommation de référence (sans modèle)
        max_signal = st.session_state.system_state["max_signal"]
        baseline_consumption = max_signal * len(signal_history)
        
        # Calculer la consommation actuelle avec le modèle
        current_consumption = sum(signal_history)
        
        # Calculer le pourcentage d'économie d'énergie
        if baseline_consumption > 0:
            energy_savings_percent = ((baseline_consumption - current_consumption) / baseline_consumption) * 100
        else:
            energy_savings_percent = 0
            
        # Récupérer le nombre d'interventions utilisateur
        interventions = st.session_state.system_state.get("total_interventions", 0)
        
        # Calculer le score de confort - Version plus sensible
        if len(signal_history) > 0:
            comfort_ratio = min(1.0, interventions / (len(signal_history) * 0.2))
            comfort_score = 100 * (1 - comfort_ratio)
        else:
            comfort_score = 100
        
        # Mettre à jour les métriques
        st.session_state.system_state["energy_metrics"].update({
            "baseline_consumption": baseline_consumption,
            "current_consumption": current_consumption,
            "energy_savings_percent": round(energy_savings_percent, 2),
            "total_interventions": interventions,
            "comfort_score": round(comfort_score, 2),
            "last_calculation": datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        st.error(f"Erreur lors du calcul des métriques: {e}")

# Fonction pour obtenir une prédiction
def get_prediction(current_value, is_intervention=False):
    if model is None:
        st.error("Modèle non disponible")
        return current_value
    
    try:
        # Si c'est une intervention, remettre à la valeur max
        if is_intervention:
            current_value = st.session_state.system_state["max_signal"]
            st.session_state.system_state["total_interventions"] += 1
        
        # Mettre à jour la step_size du modèle
        model.step_size = st.session_state.system_state["step_size"]
        
        # Obtenir la nouvelle valeur recommandée
        new_value = model.select_action(current_value)
        
        # S'assurer que la valeur ne dépasse pas max_signal
        new_value = min(new_value, st.session_state.system_state["max_signal"])
        
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
        
        # Ajouter à l'historique
        if 'history' not in st.session_state.system_state:
            st.session_state.system_state['history'] = []
        
        st.session_state.system_state['history'].insert(0, prediction_entry)
        
        # Limiter l'historique à 10 entrées
        if len(st.session_state.system_state['history']) > 10:
            st.session_state.system_state['history'].pop()
        
        # Ajouter à l'historique pour le graphique
        st.session_state.system_state["signal_history"].append(new_value)
        st.session_state.system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
        
        # Mettre à jour l'état du système
        st.session_state.system_state["current_value"] = new_value
        st.session_state.system_state["last_update"] = datetime.datetime.now().isoformat()
        
        # Mettre à jour les métriques
        calculate_energy_metrics()
        
        # Sauvegarder l'état
        save_system_state()
        
        return new_value
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return current_value

# Fonction pour générer un graphique
def generate_graph():
    signal_history = st.session_state.system_state["signal_history"]
    
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
            y=[st.session_state.system_state["max_signal"], st.session_state.system_state["max_signal"]],
            mode='lines',
            name='Valeur sans modèle',
            line=dict(color='darkgrey', width=1.5, dash='dot')
        ))
    
    # Configurer la mise en page
    fig.update_layout(
        title="Évolution du signal dans le temps",
        xaxis_title="Points de mesure",
        yaxis_title="Valeur du signal",
        yaxis=dict(range=[0, st.session_state.system_state["max_signal"] * 1.1]),
        template="plotly_white",
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=80, pad=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Ajouter une grille
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    return fig

# Fonction pour le feedback automatique
def auto_feedback(history_length, feedback_mode, no_intervention):
    if model is None:
        st.error("Modèle non disponible")
        return None
    
    try:
        # Récupérer l'historique du signal
        signal_history = st.session_state.system_state["signal_history"]
        
        # Déterminer combien de points utiliser
        if history_length == 'all':
            points_to_use = len(signal_history)
        else:
            points_to_use = min(int(history_length), len(signal_history))
        
        if points_to_use < 2:
            st.error("Pas assez de données dans l'historique pour générer un feedback")
            return None
        
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
            # Privilégier le confort - prendre une valeur entre la moyenne et le maximum
            avg_value = sum(relevant_history) / len(relevant_history)
            max_value = max(relevant_history)
            optimal_value = (avg_value + max_value) / 2
        else:
            optimal_value = st.session_state.system_state["current_value"]
        
        # Calculer une durée et un temps d'inactivité virtuels basés sur l'historique
        virtual_duration = points_to_use * st.session_state.system_state["update_interval"]
        virtual_idle_time = virtual_duration * 0.8  # 80% du temps total comme estimation
        
        # Calculer la récompense
        reward = model.calculate_reward(optimal_value, virtual_duration, virtual_idle_time)
        
        # Mettre à jour le modèle
        model.update(reward)
        
        # Extraire le nouveau triplet
        new_triplet = model.get_best_triplet().tolist()
        
        # Gérer l'intervention en fonction du paramètre no_intervention
        if not no_intervention:
            # Incrémenter le compteur d'interventions
            st.session_state.system_state["total_interventions"] = st.session_state.system_state.get("total_interventions", 0) + 1
            
            # Remettre le signal à sa valeur maximale
            st.session_state.system_state["current_value"] = st.session_state.system_state["max_signal"]
            
            # Ajouter cette valeur à l'historique du signal
            st.session_state.system_state["signal_history"].append(st.session_state.system_state["max_signal"])
            st.session_state.system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
            
            # Mettre à jour les métriques
            calculate_energy_metrics()
        
        # Sauvegarder le modèle mis à jour
        with open("model/lri_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Sauvegarder l'état du système
        save_system_state()
        
        return {
            "points_analyzed": points_to_use,
            "optimal_value": float(optimal_value),
            "reward": float(reward),
            "new_triplet": new_triplet,
            "feedback_mode": feedback_mode,
            "current_value": st.session_state.system_state["current_value"],
            "intervention": not no_intervention
        }
    
    except Exception as e:
        st.error(f"Erreur lors du feedback automatique: {e}")
        return None

# Interface principale
def main():
    # En-tête de l'application
    st.title("Contrôle de Signal LRI")
    st.markdown("*Système de contrôle adaptatif basé sur l'apprentissage par renforcement*")
    
    # Sidebar pour les contrôles
    st.sidebar.header("Paramètres du système")
    
    # Paramètres du système
    with st.sidebar.form("settings_form"):
        max_signal = st.number_input("Valeur maximale du signal", 
                                    min_value=10.0, 
                                    max_value=200.0, 
                                    value=float(st.session_state.system_state["max_signal"]),
                                    step=5.0)
        
        step_size = st.number_input("Taille du pas", 
                                   min_value=1.0, 
                                   max_value=20.0, 
                                   value=float(st.session_state.system_state["step_size"]),
                                   step=1.0)
        
        update_interval = st.number_input("Intervalle de mise à jour (secondes)", 
                                         min_value=5, 
                                         max_value=300, 
                                         value=int(st.session_state.system_state["update_interval"]),
                                         step=5)
        
        submitted = st.form_submit_button("Mettre à jour les paramètres")
        
        if submitted:
            st.session_state.system_state["max_signal"] = max_signal
            st.session_state.system_state["step_size"] = step_size
            st.session_state.system_state["update_interval"] = update_interval
            
            # Remettre le signal à sa valeur maximale lors d'une intervention
            st.session_state.system_state["current_value"] = max_signal
            st.session_state.system_state["total_interventions"] += 1
            
            # Ajouter cette valeur à l'historique
            st.session_state.system_state["signal_history"].append(max_signal)
            st.session_state.system_state["timestamp_history"].append(datetime.datetime.now().isoformat())
            
            # Sauvegarder les modifications
            save_system_state()
            
            # Recalculer les métriques
            calculate_energy_metrics()
            
            st.sidebar.success("Paramètres mis à jour. Signal remis à sa valeur maximale.")
    
    # Sidebar - Feedback automatique
    st.sidebar.header("Feedback automatique")
    with st.sidebar.form("auto_feedback_form"):
        history_options = ['10', '20', '50', '100', 'all']
        history_length = st.selectbox("Nombre de points à analyser", 
                                     options=history_options, 
                                     index=0)
        
        feedback_mode = st.selectbox("Mode de calcul", 
                                    options=['avg', 'min', 'comfort'],
                                    format_func=lambda x: {
                                        'avg': 'Moyenne', 
                                        'min': 'Minimum', 
                                        'comfort': 'Confort'
                                    }.get(x, x),
                                    index=0)
        
        no_intervention = st.checkbox("Ne pas intervenir (mode silencieux)", value=False)
        
        feedback_submitted = st.form_submit_button("Appliquer le feedback")
        
        if feedback_submitted:
            with st.spinner("Calcul du feedback en cours..."):
                result = auto_feedback(history_length, feedback_mode, no_intervention)
                if result:
                    st.sidebar.success(f"Feedback appliqué avec succès. Récompense: {result['reward']:.4f}")
                    if result['intervention']:
                        st.sidebar.info("Le signal a été remis à sa valeur maximale.")
    
    # Contenu principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Évolution du signal")
        
        # Graphique
        fig = generate_graph()
        st.plotly_chart(fig, use_container_width=True)
        
        # Boutons d'action
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("Obtenir une prédiction", key="predict_btn", use_container_width=True):
                with st.spinner("Calcul de la prédiction en cours..."):
                    current_value = st.session_state.system_state["current_value"]
                    new_value = get_prediction(current_value)
                    st.success(f"Nouvelle valeur: {new_value:.2f}")
        
        with col_btn2:
            if st.button("Intervenir (reset signal)", key="intervene_btn", use_container_width=True):
                with st.spinner("Intervention en cours..."):
                    current_value = st.session_state.system_state["current_value"]
                    new_value = get_prediction(current_value, is_intervention=True)
                    st.info(f"Signal remis à sa valeur maximale: {new_value:.2f}")
        
        with col_btn3:
            if st.button("Calculer les métriques", key="metrics_btn", use_container_width=True):
                with st.spinner("Calcul des métriques en cours..."):
                    calculate_energy_metrics()
                    st.success("Métriques mises à jour")
    
    with col2:
        st.header("Informations système")
        
        # Afficher la valeur actuelle
        st.metric(
            label="Valeur actuelle du signal",
            value=f"{st.session_state.system_state['current_value']:.2f}",
            delta=f"{st.session_state.system_state['current_value'] - st.session_state.system_state['max_signal']:.2f}"
        )
        
        # Afficher les métriques d'économie d'énergie
        energy_metrics = st.session_state.system_state["energy_metrics"]
        
        st.subheader("Métriques de performance")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric(
                label="Économie d'énergie",
                value=f"{energy_metrics['energy_savings_percent']:.2f}%"
            )
        
        with metrics_col2:
            st.metric(
                label="Score de confort",
                value=f"{energy_metrics['comfort_score']:.2f}/100"
            )
        
        st.metric(
            label="Interventions totales",
            value=st.session_state.system_state["total_interventions"]
        )
        
        # Historique des prédictions
        st.subheader("Historique des prédictions")
        history = st.session_state.system_state.get('history', [])
        
        if not history:
            st.info("Aucune prédiction dans l'historique")
        else:
            # Créer un DataFrame pour afficher l'historique
            df = pd.DataFrame(history)
            if not df.empty:
                df = df[['timestamp', 'current_value', 'new_value']]
                df.columns = ['Horodatage', 'Valeur précédente', 'Nouvelle valeur']
                st.dataframe(df, use_container_width=True)
                
    # Mode automatique
    st.header("Mode automatique")
    
    # Initialiser les variables d'état pour le mode automatique
    if 'auto_update' not in st.session_state:
        st.session_state['auto_update'] = True
        st.session_state['last_auto_update'] = None
        st.session_state['auto_update_counter'] = 0  # Pour forcer les reruns
    
    # Toggle pour la mise à jour automatique
    auto_update = st.toggle("Activer la mise à jour automatique", 
                          value=st.session_state['auto_update'],
                          key="auto_update_toggle")
    
    # Mettre à jour l'état
    if auto_update != st.session_state['auto_update']:
        st.session_state['auto_update'] = auto_update
        if auto_update:
            st.session_state['last_auto_update'] = time.time()
            st.success(f"Mise à jour automatique activée (intervalle: {st.session_state.system_state['update_interval']} secondes)")
        else:
            st.info("Mise à jour automatique désactivée")
    
    # Afficher le compteur de temps jusqu'à la prochaine mise à jour
    if auto_update:
        current_time = time.time()
        last_update = st.session_state.get('last_auto_update', current_time)
        
        if last_update is None:
            last_update = current_time
            st.session_state['last_auto_update'] = last_update
        
        # Calculer le temps écoulé et le temps restant
        elapsed = current_time - last_update
        interval = st.session_state.system_state['update_interval']
        remaining = max(0, interval - elapsed)
        
        # Afficher la progression
        progress = min(1.0, elapsed / interval)
        st.progress(progress)
        st.write(f"Prochaine mise à jour dans {remaining:.1f} secondes")
        
        # Si le temps est écoulé, faire une nouvelle prédiction
        if elapsed >= interval:
            with st.spinner("Mise à jour automatique en cours..."):
                # Réinitialiser le compteur
                st.session_state['last_auto_update'] = current_time
                
                # Effectuer la prédiction
                current_value = st.session_state.system_state["current_value"]
                new_value = get_prediction(current_value)
                
                # Message de succès
                st.success(f"Valeur mise à jour: {new_value:.2f}")
                
                # Incrémenter le compteur pour forcer le rerun
                st.session_state['auto_update_counter'] += 1
                
                # Forcer le rechargement
                st.rerun()
        else:
            # Code crucial: forcer une actualisation périodique même sans atteindre l'intervalle complet
            # Cela garantit que la barre de progression se met à jour
            st.empty()  # Placeholder pour forcer le rendu
            
            # Recharger automatiquement toutes les 2 secondes pour actualiser la barre de progression
            if 'last_refresh' not in st.session_state:
                st.session_state['last_refresh'] = time.time()
            
            refresh_elapsed = current_time - st.session_state.get('last_refresh', 0)
            if refresh_elapsed > 2:  # Rafraîchir toutes les 2 secondes
                st.session_state['last_refresh'] = current_time
                st.rerun()

    # Ajouter un bouton d'actualisation manuelle pour dépanner
    if auto_update:
        if st.button("Actualiser maintenant", key="force_refresh"):
            st.session_state['auto_update_counter'] += 1
            st.rerun()

if __name__ == "__main__":
    # Créer les dossiers nécessaires
    os.makedirs("model", exist_ok=True)
    
    # Lancer l'application
    main()
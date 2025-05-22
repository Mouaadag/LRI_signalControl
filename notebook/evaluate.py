# Créer un fichier evaluate.py dans le même répertoire que votre notebook
# filepath: /Users/mouaad/Desktop/haddam_paper/notebook/ModelTripleProba/evaluate.py
import numpy as np

from UserModel import UserModel


def evaluate_triplet(args):
    """
    Évalue un triplet de probabilités spécifique.
    """
    triplet, triplet_idx, n_executions, n_steps_per_execution, step_size = args
    p_diminuer, p_maintenir, p_augmenter = triplet
    i, j = triplet_idx
    
    print(f"Évaluation du triplet: p_dim={p_diminuer:.2f}, p_maint={p_maintenir:.2f}, p_aug={p_augmenter:.2f}")
    
    # Résultats pour ce triplet
    triplet_energy = []
    triplet_m = []
    triplet_interventions = []
    
    for execution in range(n_executions):
        # Création d'un nouveau modèle utilisateur
        user = UserModel(s_min=0, s_max=90, a0=0.2, m0=35, pre=0.35)
        
        # Variables de simulation
        current_signal = 90
        signal_history = [user.s_max]
        m_history = []
        total_interventions = 0
        
        step = 0
        while step < n_steps_per_execution:
            # Utiliser le triplet de probabilités fixe
            action_probs = np.array(triplet)
            
            # Ajuster si on est aux limites
            if current_signal <= 0 + step_size:
                action_probs[0] = 0  # Ne peut pas diminuer
            if current_signal >= 90 - step_size:
                action_probs[2] = 0  # Ne peut pas augmenter
            
            # S'assurer que toutes les probabilités sont positives
            action_probs = np.maximum(0, action_probs)
            
            # Normaliser
            sum_probs = np.sum(action_probs)
            if sum_probs > 0:
                action_probs = action_probs / sum_probs
            else:
                action_probs = np.array([0, 1, 0])
            
            # Sélectionner l'action
            action = np.random.choice(3, p=action_probs)
            
            # Appliquer l'action
            if action == 0:  # Diminuer
                new_signal = max(0, current_signal - step_size)
            elif action == 2:  # Augmenter
                new_signal = min(90, current_signal + step_size)
            else:  # Maintenir
                new_signal = current_signal
            
            # Mémoriser la valeur précédente
            old_value = signal_history[-1]
            
            # Appliquer via le modèle utilisateur
            signal_value = user.update_parameters_with_value(new_signal)
            signal_history.append(signal_value)
            
            # Enregistrer le paramètre m
            m_history.append(user.m)
            
            # Vérifier s'il y a eu intervention
            if signal_value == user.s_max and old_value != user.s_max:
                total_interventions += 1
                current_signal = 90
            else:
                current_signal = new_signal
            
            step += 1
        
        # Calculer l'énergie comme l'aire sous la courbe des valeurs du signal
        from numpy import trapezoid
        total_energy = trapezoid(signal_history) / n_steps_per_execution
        
        # Stocker les résultats de cette exécution
        triplet_energy.append(total_energy)
        triplet_m.append(sum(m_history) / len(m_history))
        triplet_interventions.append(total_interventions)
    
    # Moyennes sur toutes les exécutions
    energy_mean = sum(triplet_energy) / len(triplet_energy) if triplet_energy else 0
    m_mean = sum(triplet_m) / len(triplet_m) if triplet_m else 0
    interventions_mean = sum(triplet_interventions) / len(triplet_interventions) if triplet_interventions else 0
    
    return (i, j, energy_mean, m_mean, interventions_mean)


def evaluate_triplet_exp2(args):
    """
    Évalue un triplet de probabilités pour l'expérience 2 en utilisant la récompense mixte.
    """
    triplet, triplet_idx, n_executions, n_steps_per_execution, step_size, gamma = args
    p_diminuer, p_maintenir, p_augmenter = triplet
    t_idx = triplet_idx
    
    print(f"Évaluation du triplet {t_idx+1}: [{p_diminuer:.2f}, {p_maintenir:.2f}, {p_augmenter:.2f}]")
    
    # Résultats pour ce triplet
    triplet_rewards = []
    triplet_energy = []
    triplet_m = []
    
    for execution in range(n_executions):
        # Création d'un nouveau modèle utilisateur
        user = UserModel(s_min=0, s_max=90, a0=0.2, m0=35, pre=0.35)
        
        # Variables de simulation
        current_signal = 90
        signal_history = [user.s_max]
        total_energy = 0
        m_history = []
        
        # Compteur pour alpha_t
        time_step = 0
        
        step = 0
        while step < n_steps_per_execution:
            # Variables pour le calcul de la récompense par épisode
            episode_energy_sum = 0
            episode_steps = 0
            user_intervened = False
            
            # Continuer jusqu'à une intervention ou fin de la simulation
            while not user_intervened and step < n_steps_per_execution:
                # Utiliser le triplet de probabilités fixe
                action_probs = np.array(triplet)
                
                # Ajuster si on est aux limites
                if current_signal <= 0 + step_size:
                    action_probs[0] = 0  # Ne peut pas diminuer
                if current_signal >= 90 - step_size:
                    action_probs[2] = 0  # Ne peut pas augmenter
                
                # Normaliser
                sum_probs = np.sum(action_probs)
                if sum_probs > 0:
                    action_probs = action_probs / sum_probs
                else:
                    action_probs = np.array([0, 1, 0])
                
                # Sélectionner l'action
                action = np.random.choice(3, p=action_probs)
                
                # Appliquer l'action
                if action == 0:  # Diminuer
                    new_signal = max(0, current_signal - step_size)
                elif action == 2:  # Augmenter
                    new_signal = min(90, current_signal + step_size)
                else:  # Maintenir
                    new_signal = current_signal
                
                # Mémoriser la valeur précédente
                old_value = signal_history[-1]
                
                # Appliquer via le modèle utilisateur
                signal_value = user.update_parameters_with_value(new_signal)
                signal_history.append(signal_value)
                
                # Accumuler l'énergie
                episode_energy_sum += signal_value
                episode_steps += 1
                total_energy += signal_value
                m_history.append(user.m)
                
                # Vérifier s'il y a eu intervention
                if signal_value == user.s_max and old_value != user.s_max:
                    user_intervened = True
                    current_signal = 90
                else:
                    current_signal = new_signal
                
                step += 1
            
            # Calculer la récompense de l'épisode avec la récompense mixte
            if episode_steps > 0:
                # Calculer les composantes de base
                episode_energy_avg = episode_energy_sum / episode_steps
                energy_efficiency = (user.s_max - episode_energy_avg) / user.s_max
                comfort_ratio = (episode_steps - (1 if user_intervened else 0)) / episode_steps
                episode_reward = (1- gamma ) * energy_efficiency + gamma * comfort_ratio
                triplet_rewards.append(episode_reward)
        
        # Calculer les moyennes pour cette exécution
        triplet_energy.append(total_energy / n_steps_per_execution)
        triplet_m.append(sum(m_history) / len(m_history))
    
    # Moyennes sur toutes les exécutions
    reward = sum(triplet_rewards) / len(triplet_rewards) if triplet_rewards else 0
    energy = sum(triplet_energy) / len(triplet_energy) if triplet_energy else 0
    m_value = sum(triplet_m) / len(triplet_m) if triplet_m else 0
    
    return (t_idx, reward, energy, m_value)


def run_simulation_exp3(args):
    """
    Exécute une simulation complète pour l'expérience 3.
    Conçu pour être exécuté en parallèle.
    
    Paramètres:
    -----------
    args : tuple
        (run_id, n_steps, period_length, learning_rate, gamma, pre, step_size)
    
    Retourne:
    ---------
    tuple
        (run_id, run_probabilities, run_energy_values, run_m_values, run_intervention_counts)
    """
    import numpy as np
    from numpy import trapz

    from LRIagent import LRIAgent
    from UserModel import UserModel

    # Déballer les arguments
    run_id, n_steps, period_length, learning_rate, gamma, pre, step_size = args
    
    # Nombre de périodes
    n_periods = n_steps // period_length
    
    # Stocker les résultats pour cette exécution
    run_probabilities = np.zeros((n_periods, 3))  # 3 actions: diminuer, maintenir, augmenter
    run_energy_values = np.zeros(n_periods)
    run_m_values = np.zeros(n_periods)
    run_intervention_counts = np.zeros(n_periods)
    
    # Initialisation de l'agent LRI et du modèle utilisateur
    agent = LRIAgent(step_size=step_size, alpha=learning_rate, gamma=gamma)
    user = UserModel(s_min=0, s_max=90, a0=0.2, m0=35, pre=pre)
    
    # Variables pour la simulation
    current_signal = 90  # Signal initial
    
    # Variables pour les statistiques par période
    period_energy = []  # Stocker toutes les valeurs pour calculer l'aire
    period_m = []
    period_interventions = 0
    
    step = 0
    while step < n_steps:
        # Sélectionner une action selon les probabilités courantes
        new_signal = agent.select_action(current_signal)
        
        # Valeur avant intervention
        old_value = current_signal if step == 0 else signal_value
        
        # Appliquer l'action via le modèle utilisateur
        signal_value = user.update_parameters_with_value(new_signal)
        
        # Collecter des données
        period_energy.append(signal_value)
        period_m.append(user.m)
        
        # Durée et temps sans intervention pour calculer la récompense
        duration = 1
        idle_time = 1
        
        # Vérifier s'il y a eu intervention
        if signal_value == user.s_max and old_value != user.s_max:
            # Intervention: réinitialiser le signal et compter l'intervention
            current_signal = 90
            period_interventions += 1
            idle_time = 0
        else:
            # Pas d'intervention: mettre à jour le signal courant
            current_signal = new_signal
        
        # Calculer la récompense
        energy_efficiency = (user.s_max - signal_value) 
        comfort_ratio = idle_time
        reward = energy_efficiency * comfort_ratio ** gamma
        
        # Mise à jour de l'agent
        agent.update(reward)
        
        step += 1
        
        # Fin d'une période?
        if step % period_length == 0:
            period = step // period_length - 1
            
            # Calculer l'énergie comme l'aire sous la courbe et normaliser par la durée
            if len(period_energy) > 0:
                # Utiliser trapz pour obtenir une meilleure estimation de l'énergie totale
                total_energy = trapz(period_energy) / len(period_energy)
                run_energy_values[period] = total_energy
            else:
                run_energy_values[period] = 0
            
            # Enregistrer les autres statistiques pour cette période
            run_m_values[period] = np.mean(period_m) if period_m else 0
            run_intervention_counts[period] = period_interventions
            run_probabilities[period] = agent.get_probabilities()
            
            # Réinitialiser pour la période suivante
            period_energy = []
            period_m = []
            period_interventions = 0
            
            # Afficher progression toutes les 20 périodes
            if period % 20 == 0 and period > 0:
                print(f"  Run {run_id+1}: {period+1}/{n_periods} périodes complétées ({(period+1)/n_periods*100:.1f}%)")
    
    print(f"Exécution {run_id+1} terminée")
    return run_id, run_probabilities, run_energy_values, run_m_values, run_intervention_counts
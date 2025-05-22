import numpy as np

class LRIAgent:
    """
    Agent LRI (Linear Reward Inaction) pour exploration combinatoire.
    Sélectionne parmi une liste prédéfinie de triplets de probabilités.
    """
    
    def __init__(self, triplets=None, step_size=5, alpha=0.0005, gamma=0.45, 
                 reward_threshold=0.0, s_min=0, s_max=90, granularity=0.1):
        """
        Initialise l'agent LRI pour exploration combinatoire.
        
        Paramètres:
        -----------
        triplets : array-like
            Liste de triplets prédéfinis [p_diminuer, p_maintenir, p_augmenter]
        step_size : float
            Taille du pas d'augmentation/diminution du signal
        alpha : float
            Taux d'apprentissage pour la mise à jour des probabilités
        gamma : float
            Facteur de pondération entre économie d'énergie et confort
        reward_threshold : float
            Seuil minimal de récompense pour mettre à jour les probabilités
        s_min, s_max : float
            Valeurs minimale et maximale du signal
        granularity : float
            Granularité pour générer des triplets automatiquement si aucun n'est fourni
        """
        # Paramètres de base
        self.step_size = step_size
        self.alpha = alpha
        self.gamma = gamma
        self.reward_threshold = reward_threshold
        self.s_min = s_min
        self.s_max = s_max
        
        # Génération ou utilisation des triplets fournis
        if triplets is None:
            self.triplets = self._generate_triplets(granularity)
        else:
            self.triplets = np.array(triplets)
            
        self.n_triplets = len(self.triplets)
        print(f"Agent initialisé avec {self.n_triplets} triplets de probabilités")
        
        # Probabilités de sélection (équiprobables au départ)
        self.triplet_probs = np.ones(self.n_triplets) / self.n_triplets
        
        # Variables de suivi
        self.current_triplet_idx = None
        self.last_action = None
        self.action_history = []
        self.reward_history = []
        self.selected_triplet_history = []
        self.best_triplet_history = []
        
        # Statistiques par triplet
        self.triplet_rewards = {i: [] for i in range(self.n_triplets)}
        self.triplet_usage_count = np.zeros(self.n_triplets)
    
    def _generate_triplets(self, granularity=0.1):
        """
        Génère tous les triplets [p_diminuer, p_maintenir, p_augmenter] avec somme = 1
        """
        triplets = []
        
        for p_dim in np.arange(0, 1 + granularity/2, granularity):
            for p_aug in np.arange(0, 1 + granularity/2, granularity):
                p_maint = 1 - p_dim - p_aug
                
                # Vérifier validité (somme = 1, valeurs ≥ 0)
                if p_maint >= -granularity/2:
                    # Normaliser pour somme = 1 exactement
                    triplet = [p_dim, max(0, p_maint), p_aug]
                    if sum(triplet) > 0:  # Éviter division par zéro
                        triplet = [p/sum(triplet) for p in triplet]
                        triplets.append(triplet)
                    
        return np.array(triplets)
    
    def select_action(self, current_value):
        """
        Sélectionne une action basée sur un triplet choisi selon la distribution de probabilité.
        
        Paramètres:
        -----------
        current_value : float
            Valeur actuelle du signal
            
        Retourne:
        ---------
        float
            La nouvelle valeur du signal après application de l'action
        """
        # Sélectionner un triplet selon les probabilités actuelles
        self.current_triplet_idx = np.random.choice(self.n_triplets, p=self.triplet_probs)
        triplet = self.triplets[self.current_triplet_idx]
        p_dim, p_maint, p_aug = triplet
        
        # Enregistrer le triplet sélectionné
        self.selected_triplet_history.append(triplet)
        self.triplet_usage_count[self.current_triplet_idx] += 1
        
        # Ajuster en fonction des contraintes
        action_probs = np.array([p_dim, p_maint, p_aug])
        if current_value <= self.s_min + self.step_size:
            action_probs[0] = 0  # Ne peut pas diminuer
        if current_value >= self.s_max - self.step_size:
            action_probs[2] = 0  # Ne peut pas augmenter
            
        # Normaliser si nécessaire
        sum_probs = np.sum(action_probs)
        if sum_probs > 0:
            action_probs = action_probs / sum_probs
        else:
            action_probs = np.array([0, 1, 0])
        
        # Sélectionner l'action
        self.last_action = np.random.choice(3, p=action_probs)
        self.action_history.append(self.last_action)
        
        # Appliquer l'action
        if self.last_action == 0:  # Diminuer
            return max(self.s_min, current_value - self.step_size)
        elif self.last_action == 2:  # Augmenter
            return min(self.s_max, current_value + self.step_size)
        else:  # Maintenir
            return current_value
    
    def update(self, reward):
        """
        Met à jour les probabilités de sélection des triplets selon la récompense.
        
        Paramètres:
        -----------
        reward : float
            Récompense obtenue pour la dernière action
        """
        # Enregistrer la récompense
        self.reward_history.append(reward)
        
        # Stocker la récompense pour le triplet actuel
        self.triplet_rewards[self.current_triplet_idx].append(reward)
        
        # Mise à jour LRI si la récompense dépasse le seuil
        if reward > self.reward_threshold:
            # Vecteur unitaire pour le triplet choisi
            e_i = np.zeros(self.n_triplets)
            e_i[self.current_triplet_idx] = 1
            
            # Formule de mise à jour LRI
            self.triplet_probs = self.triplet_probs + self.alpha * reward * (e_i - self.triplet_probs)
            
            # Éviter les valeurs extrêmes et normaliser
            self.triplet_probs = np.clip(self.triplet_probs, 1e-5, 1)
            self.triplet_probs = self.triplet_probs / np.sum(self.triplet_probs)
        
        # Enregistrer le meilleur triplet actuel
        best_triplet_idx = np.argmax(self.triplet_probs)
        self.best_triplet_history.append(self.triplets[best_triplet_idx])
    
    def calculate_reward(self, v_signal, duration, idle_time):
        if duration <= 0:
            return 0
            
        # Efficacité énergétique normalisée
        energy_efficiency = (self.s_max - v_signal) / self.s_max
        
        # Ratio de confort (temps sans intervention)
        comfort_ratio = idle_time / duration
        
        # Récompense combinée avec pondération complémentaire
        reward = (1 - self.gamma) * energy_efficiency + self.gamma * comfort_ratio
        
        return reward
    
    def get_probabilities(self):
        """
        Retourne les probabilités des actions actuellement actives.
        Pour l'expérience 3, nous avons besoin de retourner les probabilités du triplet
        utilisé le plus récemment ou du meilleur triplet.

        Retourne:
        ---------
        numpy.array
            Tableau des probabilités [p_dim, p_maint, p_aug]
        """
        # Si nous avons un indice de triplet actuel, utiliser ce triplet
        if self.current_triplet_idx is not None:
            return self.triplets[self.current_triplet_idx]
        # Sinon, retourner le meilleur triplet
        elif len(self.triplet_probs) > 0:
            return self.triplets[np.argmax(self.triplet_probs)]
        # En cas d'erreur, retourner un triplet équilibré
        else:
            return np.array([1/3, 1/3, 1/3])
    
    def get_best_triplet(self):
        """Retourne le triplet avec la plus haute probabilité."""
        return self.triplets[np.argmax(self.triplet_probs)]
    
    def get_top_triplets(self, n=5):
        """Retourne les n meilleurs triplets avec leurs probabilités."""
        top_indices = np.argsort(self.triplet_probs)[::-1][:n]
        return [(self.triplets[i], self.triplet_probs[i]) for i in top_indices]
    
    def get_average_rewards(self):
        """Retourne la récompense moyenne pour chaque triplet utilisé."""
        avg_rewards = np.zeros(self.n_triplets)
        for i in range(self.n_triplets):
            if self.triplet_rewards[i]:
                avg_rewards[i] = sum(self.triplet_rewards[i]) / len(self.triplet_rewards[i])
        return avg_rewards
    
    def plot_convergence(self, window_size=100):
        """
        Visualise la convergence de l'agent vers un triplet optimal.
        
        Paramètres:
        -----------
        window_size : int
            Taille de la fenêtre pour le lissage des courbes
        """
        if not self.best_triplet_history:
            print("Aucune donnée d'historique disponible pour visualisation")
            return
            
        import matplotlib.pyplot as plt

        # Préparer les données d'historique
        history = np.array(self.best_triplet_history)
        p_dim_history = history[:, 0]
        p_maint_history = history[:, 1]
        p_aug_history = history[:, 2]
        
        # Lissage des courbes
        def smooth(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode='valid')
            return y_smooth
        
        x_smooth = np.arange(len(p_dim_history) - window_size + 1)
        p_dim_smooth = smooth(p_dim_history, window_size)
        p_maint_smooth = smooth(p_maint_history, window_size)
        p_aug_smooth = smooth(p_aug_history, window_size)
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        plt.plot(x_smooth, p_dim_smooth, label='p(diminuer)')
        plt.plot(x_smooth, p_maint_smooth, label='p(maintenir)')
        plt.plot(x_smooth, p_aug_smooth, label='p(augmenter)')
        plt.xlabel('Pas de temps')
        plt.ylabel('Probabilité')
        plt.title('Convergence vers un triplet optimal')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Afficher le triplet final
        final_triplet = self.get_best_triplet()
        print(f"Triplet final: Diminuer={final_triplet[0]:.3f}, Maintenir={final_triplet[1]:.3f}, Augmenter={final_triplet[2]:.3f}")
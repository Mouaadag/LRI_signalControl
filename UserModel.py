import numpy as np


class UserModel:
    """
    Modèle utilisateur simulé pour intervention basée sur la valeur d’un signal.
    """

    def __init__(self, s_min=0, s_max=90, a0=0.2, m0=35, pre=0.35, deterministic=False):
        self.s_min = s_min
        self.s_max = s_max
        self.a0 = a0
        self.m0 = m0
        self.pre = pre
        self.deterministic = deterministic

        self.reset()

    def reset(self):
        self.a = self.a0
        self.m = self.m0
        self.p_bar = 0
        self.eff_pas_var = 0

        self.a_history = [self.a0]
        self.m_history = [self.m0]
        self.p_bar_history = [0]
        self.signal_history = []

    def lectureProbaInterv(self, a, m, v):
        x = a * (v - m)
        if x < -100:
            return 1.0
        if x > 100:
            return 0.0
        return 1 / (1 + np.exp(x))

    def userInterv(self, a, m, v):
        p = self.lectureProbaInterv(a, m, v)
        if self.deterministic:
            return self.s_max if p > 0.5 else v
        return self.s_max if np.random.random() < p else v

    def update_parameters_with_value(self, value):
        v = max(self.s_min, min(self.s_max, value))
        p = self.lectureProbaInterv(self.a, self.m, v)

        eff_pre_val = (self.s_max - v) / (self.s_max - self.s_min)
        normalized_diff = (self.s_max - v) / (self.s_max - self.s_min)
        self.eff_pas_var = self.pre * normalized_diff + (1 - self.pre) * self.eff_pas_var

        a_inter = (eff_pre_val + self.eff_pas_var) / 2

        # Nouveau calcul stabilisé pour a
        if a_inter >= 0.98:
            self.a = 50
        else:
            self.a = 1 / (1 - a_inter)

        self.p_bar = self.pre * p + (1 - self.pre) * self.p_bar
        self.m = self.m0 + self.p_bar * (self.s_max - self.m0)

        new_v = self.userInterv(self.a, self.m, v)

        # Historique enrichi
        self.a_history.append(self.a)
        self.m_history.append(self.m)
        self.p_bar_history.append(self.p_bar)
        self.signal_history.append({
            "v": v,
            "p": p,
            "new_v": new_v,
            "a": self.a,
            "m": self.m,
            "p_bar": self.p_bar
        })

        return new_v
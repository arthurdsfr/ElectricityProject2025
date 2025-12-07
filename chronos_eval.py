import numpy as np

#Script contenant les fonctions d'évaluation des performances

#évaluation des performances (MASE)
def calculate_mase(y_true, y_pred, y_hist, seasonality=1):
    """
    Calcule la MASE (Mean Absolute Scaled Error).
    
    Args:
        y_true (np.array): Vraies valeurs du futur (Test set).
        y_pred (np.array): Prédictions (Zero-Shot ou Fine-Tuned).
        y_hist (np.array): Historique complet (pour calculer l'échelle).
        seasonality (int): Valeur reprise pour le modèle naïf (1 pour jour d'avant).
    
    Returns:
        float: Le score MASE.
    """
    #calcul de l'erreur absolue de la prédiction (Numérateur)
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    
    #valcul de l'erreur absolue moyenne du modèle naïf sur l'historique (Dénominateur)
    #on compare chaque jour t avec t-1 sur tout l'historique
    naive_errors = np.abs(y_hist[seasonality:] - y_hist[:-seasonality])
    mae_naive = np.mean(naive_errors)
    
    #gestion du cas improbable où l'erreur naïve est 0 (division par zéro)
    if mae_naive == 0:
        return np.inf
        
    #calcul final
    return mae_forecast / mae_naive


def calculate_mda(y_true, y_pred, last_history_value):
    """
    Calcule la MDA (Mean Directional Accuracy).
    
    Args:
        y_true (np.array): Vraies valeurs (futur).
        y_pred (np.array): Valeurs prédites.
        last_history_value (float): Dernière valeur connue avant la prédiction
                                    (le prix d'aujourd'hui).
    
    Returns:
        float: Pourcentage de directions (augmentation ou baisse) correctes (entre 0.0 et 1.0).
    """
    #on construit le vecteur des valeurs précédentes (t-1)
    #pour le 1er jour prédit, le "précédent" est last_history_value
    #pour le 2ème jour prédit, le "précédent" est le vrai prix du 1er jour, etc.
    y_true_prev = np.insert(y_true[:-1], 0, last_history_value)
    
    #calcul des directions réelles (1 si hausse, -1 si baisse, 0 si stable)
    #on regarde si le prix réel a monté ou baissé par rapport à la veille
    actual_direction = np.sign(y_true - y_true_prev)
    
    #valcul des directions prédites (1 si hausse, -1 si baisse, 0 si stable)
    #on regarde si le modèle a prédit une hausse ou une baisse par rapport à la veille
    predicted_direction = np.sign(y_pred - y_true_prev)
    
    #comparaison des directions
    #on vérifie si (Haut == Haut) ou (Bas == Bas)
    correct_predictions = (actual_direction == predicted_direction)
    
    #moyenne (pourcentage de réussite)
    return np.mean(correct_predictions)


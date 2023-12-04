import pickle
import numpy as np

def alias_setup(probs):
    K = len(probs)
    q = [0.0] * K
    J = [0] * K
    
    # Étape 1: Initialiser les listes small et large
    smaller = []
    larger = []
    for k, prob in enumerate(probs):
        q[k] = K * prob
        if q[k] < 1.0:
            smaller.append(k)
        else:
            larger.append(k)
    
    # Étape 2: Remplir les compartiments
    while smaller and larger:
        small, large = smaller.pop(), larger.pop()
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    
    return J, q

def alias_draw(J, q):
    K = len(J)
    # Choisissez un compartiment au hasard
    kk = int(np.floor(np.random.rand() * K))
    # Renvoie kk avec probabilité q[kk], sinon renvoie J[kk]
    return kk if np.random.rand() < q[kk] else J[kk]



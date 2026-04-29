import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Étape 1 : Génération du Dataset
# 1. Générer 2000 points aléatoires (x, y) entre -5 et 5
raw_inputs = np.random.uniform(-5, 5, (2000, 2))
 
# 2. Fonction cible f(x, y)
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)

x_coords = raw_inputs[:, 0]
y_coords = raw_inputs[:, 1]
z_raw = f(x_coords, y_coords)

print(x_coords, y_coords)
# 3. Normalisation
# Entrées (x, y) ramenées à [-1, 1]
inputs_norm = raw_inputs / 5.0


# Sortie (z) normalisée entre [0, 1] (Min-Max Scaling)
z_min, z_max = np.min(z_raw), np.max(z_raw)
z_norm = (z_raw - z_min) / (z_max - z_min)
z_norm = z_norm.reshape(-1, 1) # Format (2000, 1) pour le réseau

# Étape 2 : Architecture du Réseau (MLP)
class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        
        for i in range(len(layers) - 1):
            # He Initialization : sqrt(2/n_in)
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.a = [x]
        self.z_list = []

        activation = x
        # Parcours des couches cachées
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_list.append(z)
            activation = self.relu(z)
            self.a.append(activation)
        
        # Couche de sortie (Linéaire pour la régression)
        output = np.dot(activation, self.weights[-1]) + self.biases[-1]
        self.a.append(output)
        return output

    def compute_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    # --- Étape 3 : Backpropagation ---
    def backward(self, y_true, y_pred, lr):
        m = y_true.shape[0]
        
        # 1. Calcul du gradient de la perte par rapport à la sortie
        dz = y_pred - y_true 
        
        # 2. mise à jour des poids de la dernière couche à la première
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                #différentiel de la fonction d'activation ReLU
                dz = np.dot(dz, self.weights[i].T) * (self.z_list[i-1] > 0)
            
            # mise à jour des poids et des biais
            self.weights[i] = self.weights[i] - lr * dw
            self.biases[i] = self.biases[i] - lr * db

# Exécution du Modèle


# Initialisation du modèle [2, 64, 64, 1]
mlp_model = MLP(layers=[2, 64, 64, 1])

# Test du Forward pass avec les données générées
predictions = mlp_model.forward(inputs_norm)

# Calcul du coût initial (Loss)
initial_loss = mlp_model.compute_mse(z_norm, predictions)

print(f"Dataset généré avec {inputs_norm.shape[0]} points.")
print(f"Architecture du réseau : {mlp_model.layers}")
print(f"MSE Loss initial (avant entraînement) : {initial_loss:.6f}")

# --- Visualisation de la Vérité Terrain ---
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_coords, y_coords, z_raw, c=z_raw, cmap='viridis')
plt.colorbar(scatter, label='Altitude Z')
ax.set_title("Visualisation 3D du Dataset (Vérité Terrain)")
plt.show()

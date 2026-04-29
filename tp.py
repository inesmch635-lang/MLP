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


# 3. Normalisation
# Entrées (x, y) ramenées à [-1, 1]
inputs_norm = raw_inputs / 5.0


# Sortie (z) normalisée entre [0, 1] (Min-Max Scaling)
z_min, z_max = np.min(z_raw), np.max(z_raw)
z_norm = (z_raw - z_min) / (z_max - z_min)
z_norm = z_norm.reshape(-1, 1) # Format (2000, 1) pour le réseau

# Étape 2 : Architecture du Réseau (MLP)
class MLP:
    def __init__(self, layers, gamma=0.9):
        self.layers = layers
        self.gamma = gamma # Paramètre du Momentum (0 pour SGD standard, 0.9 pour Momentum)
        self.weights = []
        self.biases = []
        self.v_weights = [] # Initialisation des vecteurs de vitesse pour le Momentum
        self.v_biases = []
        
        
        for i in range(len(layers) - 1):
            # He Initialization : sqrt(2/n_in)
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            # Les vitesses sont initialisées à zéro
            self.v_weights.append(np.zeros_like(w))
            self.v_biases.append(np.zeros_like(b))

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
            

            # --- Mise à jour avec Momentum ---
            # Si gamma=0, cela revient exactement à la descente de gradient standard
            self.v_weights[i] = self.gamma * self.v_weights[i] + lr * dw
            self.v_biases[i] = self.gamma * self.v_biases[i] + lr * db

            # mise à jour des poids et des biais
            self.weights[i] = self.weights[i] - lr * dw
            self.biases[i] = self.biases[i] - lr * db

# Exécution du Modèle


# Initialisation du modèle [2, 64, 64, 1]
mlp_model = MLP(layers=[2, 64, 64, 1], gamma=0.9)

# Test du Forward pass avec les données générées
predictions = mlp_model.forward(inputs_norm)

# Calcul du coût initial (Loss)
initial_loss = mlp_model.compute_mse(z_norm, predictions)

print(f"Dataset généré avec {inputs_norm.shape[0]} points.")
print(f"Architecture du réseau : {mlp_model.layers}")
print(f"MSE Loss initial (avant entraînement) : {initial_loss:.6f}")

# --- Étape 4 : Entraînement ---
mlp_model = MLP(layers=[2, 64, 64, 1],gamma=0.9)
epochs = 1000
learning_rate = 0.01
losses = []

print(f"Start Training...(Gamma={mlp_model.gamma})")
for epoch in range(epochs):
    # Forward pass
    predictions = mlp_model.forward(inputs_norm)
    
    # Calcul du MSE
    loss = np.mean((z_norm - predictions)**2)
    losses.append(loss)
    
    # Backward pass (mise à jour des poids)
    mlp_model.backward(z_norm, predictions, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} -> Loss: {loss:.6f}")

# --- Visualisation Final  ---

# 1. Courbe de la perte (Loss) au fil des epochs
plt.figure(figsize=(8, 5))
plt.plot(losses,color='blue', linewidth=2)
plt.title("Évolution de la Perte (MSE) vs Époques")
plt.xlabel("Époques")
plt.ylabel("MSE")
plt.grid(True, alpha=0.3)

# 2. Surface prédite par le MLP vs la vérité terrain
grid = np.linspace(-5, 5, 40)
gx, gy = np.meshgrid(grid, grid)
test_inputs = np.c_[gx.ravel(), gy.ravel()]
test_norm = test_inputs / 5.0

pred_norm = mlp_model.forward(test_norm)
# Denormalization des prédictions
pred_final = pred_norm * (z_max - z_min) + z_min
gz_pred = pred_final.reshape(gx.shape)

fig = plt.figure(figsize=(14, 7))
# Vérité terrain
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(gx, gy, f(gx, gy), cmap='viridis',alpha=0.8)
ax1.set_title("Ground Truth (Fonction Réelle)")

# Prédiction du MLP
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(gx, gy, gz_pred, cmap='magma',alpha=0.8)
ax2.set_title(f"MLP Prediction (Gamma={mlp_model.gamma})")

plt.tight_layout()
plt.show()

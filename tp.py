import numpy as np
import mpl_toolkits.mplot3d.axes3d as Axes3d
import matplotlib.pyplot as plt
#  Génération du Dataset 
inputs= 10*np.random.random_sample((2000, 2))-5 # 2000 échantillons, 2 caractéristiques
print(inputs)
def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))+0.5*np.cos(2*x+2*y)


x=inputs[:,0]
y=inputs[:,1]
z=f(x,y)


# تسوية المدخلات (x, y) من نطاق [-5, 5] إلى [-1, 1]
inputs_norm = inputs / 5.0 

# تسوية المخرجات (z) 
z_min, z_max = np.min(z), np.max(z)
z_norm = (z - z_min) / (z_max - z_min) # تجعل القيم بين 0 و 1

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# رسم النقاط مع تلوينها بناءً على الارتفاع (c=z_real)
scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
plt.colorbar(scatter)
ax.set_title("Vérité Terrain (La Carte de l'Île)")
plt.show()




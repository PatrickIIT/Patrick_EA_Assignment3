import numpy as np
import matplotlib.pyplot as plt
from eikonal_model import Eikonal2DnetCV2
from sklearn.metrics import mean_squared_error

# Generate synthetic data
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.linspace(0, 1, 100).reshape(-1, 1)
x, y = np.meshgrid(x, y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
T = np.minimum(np.sqrt(x**2 + y**2), 0.7 * np.sqrt((x-1)**2 + (y-1)**2))

# Sparse sampling
N_data = 30
idx = np.random.choice(len(x), N_data, replace=False)
x_e = x[idx].reshape(-1, 1)
y_e = y[idx].reshape(-1, 1)
T_e = T[idx].reshape(-1, 1)

# Define layers
layers = [2, 20, 20, 1]

# Train Model 1 (data-driven)
try:
    model1 = Eikonal2DnetCV2(x, y, x_e, y_e, T_e, layers, model_type="data")
    model1.train_Adam(nIter=1000)
    model1.save_results("model1_results.pkl")
    T_star_model1 = model1.predict(x, y)
except Exception as e:
    print(f"Error training Model 1: {e}")

# Train Model 2 (physics-informed)
try:
    model2 = Eikonal2DnetCV2(x, y, x_e, y_e, T_e, layers, model_type="physics")
    model2.train_Adam(nIter=1000)
    model2.save_results("model2_results.pkl")
    T_star_model2 = model2.predict(x, y)
except Exception as e:
    print(f"Error training Model 2: {e}")

# Compute RMSE
rmse_model1 = np.sqrt(mean_squared_error(T.reshape(-1), T_star_model1.reshape(-1)))
rmse_model2 = np.sqrt(mean_squared_error(T.reshape(-1), T_star_model2.reshape(-1)))

# Visualization
plt.figure(figsize=(15, 5))

# True Activation Map
plt.subplot(1, 3, 1)
plt.contourf(x.reshape(100, 100), y.reshape(100, 100), T.reshape(100, 100), levels=20, cmap='viridis')
plt.colorbar(label='Activation Time')
plt.title('True Activation Times')
plt.xlabel('x')
plt.ylabel('y')

# Predicted Activation Map (Model 1)
plt.subplot(1, 3, 2)
plt.contourf(x.reshape(100, 100), y.reshape(100, 100), T_star_model1.reshape(100, 100), levels=20, cmap='viridis')
plt.colorbar(label='Activation Time')
plt.title('Predicted Activation Times (Model 1)')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_e, y_e, c='black', s=10, label='Sampling Points')
plt.legend()

# Predicted Activation Map (Model 2)
plt.subplot(1, 3, 3)
plt.contourf(x.reshape(100, 100), y.reshape(100, 100), T_star_model2.reshape(100, 100), levels=20, cmap='viridis')
plt.colorbar(label='Activation Time')
plt.title('Predicted Activation Times (Model 2)')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_e, y_e, c='black', s=10, label='Sampling Points')
plt.legend()

plt.tight_layout()
plt.savefig('activation_maps.png')
plt.show()

# Error Plot (RMSE)
plt.figure(figsize=(8, 5))
plt.bar(['Model 1', 'Model 2'], [rmse_model1, rmse_model2])
for i, v in enumerate([rmse_model1, rmse_model2]):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.title('RMSE Error')
plt.ylabel('RMSE')
plt.savefig('rmse_error.png')
plt.show()

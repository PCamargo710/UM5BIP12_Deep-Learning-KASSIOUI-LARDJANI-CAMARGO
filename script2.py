import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
# Chargement du fichier tsv
df = pd.read_csv('TCGA_IDC_ILC.tsv', sep='\t', index_col=0)

# Création des labels (ILC vs IDC) basés sur CDH1 (perte d'expression dans l'ILC)
y = (df['CDH1'] < df['CDH1'].median()).astype(int) # 1 = ILC, 0 = IDC
X = df.values

# Normalisation (Essentiel pour le Deep Learning)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Conversion en Tenseurs PyTorch
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test.values).unsqueeze(1)

# ==========================================
# 2. AUTOENCODEUR (Extraction de Signatures)
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Compression vers un espace latent de 32 dimensions (la signature compressée)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Entraînement de l'Autoencodeur
ae_model = Autoencoder(X_scaled.shape[1])
ae_criterion = nn.MSELoss()
ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

ae_losses = []
for epoch in range(100):
    output = ae_model(X_train_t)
    loss = ae_criterion(output, X_train_t)
    ae_optimizer.zero_grad()
    loss.backward()
    ae_optimizer.step()
    ae_losses.append(loss.item())

# GRAPHE 1 : Loss de l'Autoencodeur
plt.figure(figsize=(8, 5))
plt.plot(ae_losses, color='blue', label='Reconstruction Loss')
plt.title("Apprentissage de l'Autoencodeur : Compression des Signatures")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

# ==========================================
# 3. MLP (Classification ILC vs IDC)
# ==========================================
class BreastCancerMLP(nn.Module):
    def __init__(self, input_dim):
        super(BreastCancerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

mlp_model = BreastCancerMLP(X_scaled.shape[1])
mlp_criterion = nn.BCELoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

mlp_losses = []
mlp_acc = []

for epoch in range(100):
    preds = mlp_model(X_train_t)
    loss = mlp_criterion(preds, y_train_t)
    
    mlp_optimizer.zero_grad()
    loss.backward()
    mlp_optimizer.step()
    
    # Calcul Accuracy
    with torch.no_grad():
        train_preds = (preds > 0.5).float()
        accuracy = (train_preds == y_train_t).sum() / y_train_t.shape[0]
        mlp_acc.append(accuracy.item())
        mlp_losses.append(loss.item())

# GRAPHE 2 : Courbes d'apprentissage du MLP
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (BCE)', color='red')
ax1.plot(mlp_losses, color='red', label='Loss')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='green')
ax2.plot(mlp_acc, color='green', label='Accuracy')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("Performance du MLP : Classification IDC vs ILC")
plt.show()

# ==========================================
# 4. EXTRACTION DE LA SIGNATURE (Interprétabilité)
# ==========================================
# On regarde quels gènes ont les poids les plus forts dans la première couche du MLP
weights = mlp_model.model[0].weight.abs().detach().numpy().mean(axis=0)
top_genes_idx = np.argsort(weights)[-15:]
top_genes_names = df.columns[top_genes_idx]

# GRAPHE 3 : Les gènes constituant la signature selon le Deep Learning
plt.figure(figsize=(10, 6))
plt.barh(top_genes_names, weights[top_genes_idx], color='teal')
plt.title("Signature Transcriptomique identifiée par le MLP")
plt.xlabel("Importance du Gène (Poids moyens du réseau)")
plt.show()
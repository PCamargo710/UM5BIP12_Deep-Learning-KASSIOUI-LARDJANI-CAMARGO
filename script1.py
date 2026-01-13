import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION
# ==========================================
# Chargement du fichier tsv
df = pd.read_csv('TCGA_IDC_ILC.tsv', sep='\t', index_col=0)

# Création des labels basés sur CDH1 (Marqueur clé de l'ILC selon Ciriello et al.)
# Nous utilisons la médiane comme seuil de séparation
y = np.where(df['CDH1'] < df['CDH1'].median(), 'ILC', 'IDC')

# ==========================================
# 2. GRAPHE 1 : PCA (Analyse Globale)
# ==========================================
plt.figure(figsize=(10, 7))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1', s=60)
plt.title("1. PCA : Séparation Transcriptomique IDC vs ILC")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.savefig('1_PCA_Separation.png', dpi=300)
plt.show()
plt.close()

# ==========================================
# 3. GRAPHE 2 : IMPORTANCE DES GÈNES (Signature)
# ==========================================
# On entraîne un modèle pour trouver les gènes les plus discriminants
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df, y)

importances = pd.Series(rf.feature_importances_, index=df.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title("2. Signature : Top 15 gènes discriminants")
plt.xlabel("Score d'Importance")
plt.savefig('2_Signature_Genes.png', dpi=300)
plt.show()
plt.close()

# ==========================================
# 4. GRAPHE 3 : BOXPLOTS (Validation Biologique)
# ==========================================
# On sélectionne CDH1 et FOXA1 (gènes majeurs du projet)
genes_to_plot = ['CDH1', 'FOXA1']
df_melt = df[genes_to_plot].copy()
df_melt['Type'] = y
df_plot = df_melt.melt(id_vars='Type', var_name='Gène', value_name='Expression')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_plot, x='Gène', y='Expression', hue='Type', palette='Pastel1')
plt.title("3. Validation de l'expression des gènes pivots")
plt.savefig('3_Validation_Boxplot.png', dpi=300)
plt.show()
plt.close()

# ==========================================
# 5. GRAPHE 4 : HEATMAP (Co-expression)
# ==========================================
# On regarde comment les 10 meilleurs gènes de la signature corrèlent entre eux
top_10 = importances.head(10).index
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_10].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("4. Corrélation entre les gènes de la signature")
plt.savefig('4_Heatmap_Correlation.png', dpi=300)
plt.show()
plt.close()

print("Analyse terminée. Les 4 graphiques ont été enregistrés dans votre dossier.")
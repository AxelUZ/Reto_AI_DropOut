import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Ruta al archivo de datos
file_path = '/Users/axel/Documents/Reto_AI_DropOut/students_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Seleccionar las columnas relevantes para el modelo de regresión
model_columns = [
    'average.first.period', 'failed.subject.first.period', 'dropped.subject.first.period',
    'socioeconomic.level', 'social.lag', 'scholarship.perc', 'loan.perc',
    'age', 'gender', 'admission.test', 'english.evaluation', 'general.math.eval', 'foreign'
]

# Filtrar los datos
X = data[model_columns].copy()
y = data['dropout.semester']

# Convertir columnas categóricas a numéricas
X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})
X['socioeconomic.level'] = X['socioeconomic.level'].map({
    'No information': np.nan, 'Level 1': 1, 'Level 2': 2, 'Level 3': 3,
    'Level 4': 4, 'Level 5': 5, 'Level 6': 6, 'Level 7': 7
})
X['social.lag'] = X['social.lag'].map({
    'No information': np.nan, 'Low': 1, 'Medium': 2, 'High': 3
})
X['general.math.eval'].replace({'Does not apply': np.nan, 'No information': np.nan}, inplace=True)
X['general.math.eval'] = pd.to_numeric(X['general.math.eval'], errors='coerce')
X.replace(['No information', 'Does not apply', 'Does not apply '], np.nan, inplace=True)
X['admission.test'] = pd.to_numeric(X['admission.test'], errors='coerce')
X['foreign'] = X['foreign'].map({'Local': 0, 'Yes: National': 1, 'Yes: Foreigner': 2})

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Entrenar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_imputed, y)

# Importancia de características
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

# Visualización de importancia de características
colors = ['blue' if feature != 'foreign' else 'red' for feature in feature_importances.index]
feature_importances.plot(kind='bar', figsize=(12, 6), color=colors)
plt.title('Importancia de Características')
plt.ylabel('Importancia')
plt.xlabel('Características')
plt.show()

# Análisis de Clúster
features_for_clustering = model_columns

# Preparación de los datos
X_clustering = data[features_for_clustering].copy()
X_clustering.replace(['No information', 'Does not apply', 'Does not apply '], np.nan, inplace=True)
X_clustering['socioeconomic.level'] = X_clustering['socioeconomic.level'].map({
    'No information': np.nan, 'Level 1': 1, 'Level 2': 2, 'Level 3': 3,
    'Level 4': 4, 'Level 5': 5, 'Level 6': 6, 'Level 7': 7
})
X_clustering['social.lag'] = X_clustering['social.lag'].map({
    'No information': np.nan, 'Low': 1, 'Medium': 2, 'High': 3
})
X_clustering['general.math.eval'].replace({'Does not apply': np.nan, 'No information': np.nan}, inplace=True)
X_clustering['general.math.eval'] = pd.to_numeric(X_clustering['general.math.eval'], errors='coerce')
X_clustering['admission.test'] = pd.to_numeric(X_clustering['admission.test'], errors='coerce')
X_clustering['gender'] = X_clustering['gender'].map({'Male': 1, 'Female': 0})
X_clustering['foreign'] = X_clustering['foreign'].map({'Local': 0, 'Yes: National': 1, 'Yes: Foreigner': 2})

# Imputación de valores faltantes
X_clustering_imputed = imputer.fit_transform(X_clustering)

# Realizar el análisis de clúster
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_clustering_imputed)

# Añadir la columna de clústeres a los datos originales
data['cluster'] = clusters

# Visualización de clústeres (usando scatterplot para evitar errores con variables categóricas)
num_vars = ['average.first.period', 'failed.subject.first.period', 'socioeconomic.level']
sns.pairplot(data, hue='cluster', vars=num_vars, diag_kind='hist')
plt.show()

# Análisis de la proporción de foráneos en cada clúster
cluster_foreign_proportion = data.groupby(['cluster', 'foreign']).size().unstack().fillna(0)
cluster_foreign_proportion = cluster_foreign_proportion.div(cluster_foreign_proportion.sum(axis=1), axis=0)

# Visualización de la proporción de foráneos en cada clúster
cluster_foreign_proportion.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Proporción de Estudiantes Foráneos en Cada Clúster')
plt.ylabel('Proporción')
plt.xlabel('Clúster')
plt.show()
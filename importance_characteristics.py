import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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

# Convertir la nueva variable 'foreign' a numérica
X['foreign'] = X['foreign'].map({
    'Local': 0, 'Yes: National': 1, 'Yes: Foreigner': 2
})

# Reemplazar valores no numéricos con NaN en todas las columnas
X.replace(['No information', 'Does not apply', 'Does not apply '], np.nan, inplace=True)

# Convertir admission.test a numérico
X['admission.test'] = pd.to_numeric(X['admission.test'], errors='coerce')

# Imputar valores faltantes con la media (o podrías usar otra estrategia de imputación)
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


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Ruta al archivo de datos
file_path = '/Users/axel/Documents/Reto_AI_DropOut/students_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Seleccionar las columnas relevantes para el modelo de regresión, incluyendo 'foreign'
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
X['admission.test'] = pd.to_numeric(X['admission.test'], errors='coerce')

# Convertir la columna 'foreign' a numérica
X['foreign'] = X['foreign'].map({'Local': 0, 'Yes: National': 1, 'Yes: Foreigner': 2})

# Reemplazar valores no numéricos con NaN en todas las columnas
X.replace(['No information', 'Does not apply', 'Does not apply '], np.nan, inplace=True)

# Imputar valores faltantes con la media (o podrías usar otra estrategia de imputación)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Ruta al archivo de datos
file_path = '/Users/axel/Documents/Reto_AI_DropOut/students_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Crear una columna 'dropout_event' donde 1 indica deserción y 0 indica que no
data['dropout_event'] = (data['dropout.semester'] > 0).astype(int)

# Verificar la distribución de clases antes de dividir los datos
print("Distribución de clases en los datos originales:")
print(data['dropout_event'].value_counts())

# Seleccionar las columnas relevantes para el modelo
model_columns = [
    'average.first.period', 'failed.subject.first.period', 'dropped.subject.first.period',
    'socioeconomic.level', 'social.lag', 'scholarship.perc', 'loan.perc',
    'age', 'gender', 'admission.test', 'english.evaluation', 'general.math.eval', 'foreign'
]

# Filtrar los datos
X = data[model_columns].copy()
y = data['dropout_event']

# Convertir columnas categóricas a numéricas
X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})
X['socioeconomic.level'] = X['socioeconomic.level'].map({
    'No information': np.nan, 'Level 1': 1, 'Level 2': 2, 'Level 3': 3,
    'Level 4': 4, 'Level 5': 5, 'Level 6': 6, 'Level 7': 7
})
X['social.lag'] = X['social.lag'].map({
    'No information': np.nan, 'Low': 1, 'Medium': 2, 'High': 3
})
X['foreign'] = X['foreign'].map({
    'Local': 0, 'Yes: National': 1, 'Yes: Foreigner': 2
})
X['general.math.eval'].replace({'Does not apply': np.nan, 'No information': np.nan}, inplace=True)
X['general.math.eval'] = pd.to_numeric(X['general.math.eval'], errors='coerce')
X.replace(['No information', 'Does not apply', 'Does not apply '], np.nan, inplace=True)
X['admission.test'] = pd.to_numeric(X['admission.test'], errors='coerce')

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42, stratify=y)

# Verificar la distribución de clases en los conjuntos de entrenamiento y prueba
print("Distribución de clases en el conjunto de entrenamiento:")
print(pd.Series(y_train).value_counts())
print("Distribución de clases en el conjunto de prueba:")
print(pd.Series(y_test).value_counts())

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

# Calcular y mostrar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy:.2f}")

# Opcional: Ver importancia de características
feature_importances = pd.DataFrame({'Feature': model_columns, 'Importance': np.abs(model.coef_[0])}).sort_values(by='Importance', ascending=False)
print(feature_importances)

# Visualizar importancia de características
plt.figure(figsize=(12, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Importancia de Características en el Modelo de Regresión Logística')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.show()

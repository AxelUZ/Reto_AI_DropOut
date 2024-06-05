import pandas as pd

# Ruta al archivo de datos
file_path = '/Users/axel/Documents/Reto_AI_DropOut/students_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Seleccionar las columnas relevantes para el modelo de regresión
model_columns = [
    'average.first.period', 'failed.subject.first.period', 'dropped.subject.first.period',
    'socioeconomic.level', 'social.lag', 'scholarship.perc', 'loan.perc',
    'age', 'gender', 'admission.test', 'english.evaluation', 'general.math.eval'
]

# Filtrar los datos
X = data[model_columns].copy()

# Mostrar los valores únicos de cada columna
for column in X.columns:
    print(f"Column: {column}")
    print(X[column].unique())
    print("\n")


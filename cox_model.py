import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# Ruta al archivo de datos
file_path = '/Users/axel/Documents/Reto_AI_DropOut/students_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Crear una columna 'dropout_event' donde 1 indica deserción y 0 indica que no
data['dropout_event'] = data['dropout.semester'].notna().astype(int)

# Reemplazar los NaN en 'dropout.semester' por el último semestre posible (o un valor alto)
max_semester = data['dropout.semester'].max()
data['dropout.semester'].fillna(max_semester + 1, inplace=True)

# Preparar los datos para el modelo de Cox
data_cox = data[['dropout.semester', 'dropout_event', 'average.first.period', 'failed.subject.first.period',
                 'dropped.subject.first.period', 'socioeconomic.level', 'social.lag', 'scholarship.perc',
                 'loan.perc', 'age', 'gender', 'admission.test', 'english.evaluation', 'general.math.eval', 'foreign']].copy()

# Convertir las variables categóricas a numéricas
data_cox['gender'] = data_cox['gender'].map({'Male': 1, 'Female': 0})
data_cox['socioeconomic.level'] = data_cox['socioeconomic.level'].map({
    'No information': np.nan, 'Level 1': 1, 'Level 2': 2, 'Level 3': 3,
    'Level 4': 4, 'Level 5': 5, 'Level 6': 6, 'Level 7': 7
})
data_cox['social.lag'] = data_cox['social.lag'].map({
    'No information': np.nan, 'Low': 1, 'Medium': 2, 'High': 3
})
data_cox['general.math.eval'].replace({'Does not apply': np.nan, 'No information': np.nan}, inplace=True)
data_cox['general.math.eval'] = pd.to_numeric(data_cox['general.math.eval'], errors='coerce')
data_cox['admission.test'] = pd.to_numeric(data_cox['admission.test'], errors='coerce')
data_cox['foreign'] = data_cox['foreign'].map({'Local': 0, 'Yes: National': 1, 'Yes: Foreigner': 2})

# Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
data_cox_imputed = pd.DataFrame(imputer.fit_transform(data_cox), columns=data_cox.columns)

# Ajustar el modelo de Cox
cph = CoxPHFitter()
cph.fit(data_cox_imputed, duration_col='dropout.semester', event_col='dropout_event')

# Mostrar el resumen del modelo
cph.print_summary()

# Visualizar el modelo
cph.plot()
plt.title('Coeficientes del Modelo de Cox')
plt.show()

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter
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

# Imputar valores faltantes con la media para variables numéricas
numeric_cols = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Agrupar la columna 'average.first.period'
bins = [0, 60, 70, 80, 90, 100]
labels = ['0-60', '61-70', '71-80', '81-90', '91-100']
data['average.first.period.grouped'] = pd.cut(data['average.first.period'], bins=bins, labels=labels, right=False)

# Función para plotear las curvas de supervivencia
def plot_survival(data, time_col, event_col, group_col, title):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))

    for name, grouped_df in data.groupby(group_col):
        kmf.fit(grouped_df[time_col], grouped_df[event_col], label=name)
        kmf.plot_survival_function(ci_show=True)

    plt.title(title)
    plt.xlabel('Semestre')
    plt.ylabel('Proporción de Estudiantes que no han Desertado')
    plt.show()

# Graficar curvas de supervivencia para la variable 'foreign'
plot_survival(data, 'dropout.semester', 'dropout_event', 'foreign', 'Curva de Supervivencia por Foreign')

# Graficar curvas de supervivencia para diferentes variables
variables_to_plot = [
    ('gender', 'Curva de Supervivencia por gender'),
    ('social.lag', 'Curva de Supervivencia por social.lag'),
    ('socioeconomic.level', 'Curva de Supervivencia por socioeconomic.level'),
    ('age', 'Curva de Supervivencia por age'),
    ('loan.perc', 'Curva de Supervivencia por loan.perc'),
    ('scholarship.perc', 'Curva de Supervivencia por scholarship.perc'),
    ('average.first.period.grouped', 'Curva de Supervivencia por average.first.period.grouped')
]

for group_col, title in variables_to_plot:
    plot_survival(data, 'dropout.semester', 'dropout_event', group_col, title)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo de datos
file_path = '/Users/axel/Documents/Reto_AI_DropOut/students_data.csv'

# Cargar los datos
data = pd.read_csv(file_path)


# Agregar análisis descriptivo para la variable 'foreign'
def descriptive_analysis(data):
    desc_stats = data.describe(include='all')
    print("Estadísticas Descriptivas:")
    print(desc_stats)

    # Análisis de la variable 'foreign'
    foreign_counts = data['foreign'].value_counts()
    print("\nConteo de 'foreign':")
    print(foreign_counts)

    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='foreign')
    plt.title('Distribución de Estudiantes Foráneos y Locales')
    plt.xlabel('Foreign')
    plt.ylabel('Count')
    plt.show()

    # Relaciones entre 'foreign' y otras variables
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data, x='foreign', y='average.first.period')
    plt.title('Promedio del Primer Periodo por Foreign')
    plt.xlabel('Foreign')
    plt.ylabel('Promedio del Primer Periodo')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.boxplot(data=data, x='foreign', y='dropout.semester')
    plt.title('Semestre de Deserción por Foreign')
    plt.xlabel('Foreign')
    plt.ylabel('Semestre de Deserción')
    plt.show()


# Ejecutar el análisis descriptivo
descriptive_analysis(data)

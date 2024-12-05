import pandas as pd

# Cargar el conjunto de datos
df = pd.read_csv('diabetes_dataset00.csv')
#
# Lista de columnas a eliminar
#columnas_a_eliminar = [
 #   'Pulmonary Function',
 #   'Cystic Fibrosis Diagnosis',
 #   'Neurological Assessments',
 #   'Digestive Enzyme Levels',
 #   'Liver Function Tests',
 #   'Socioeconomic Factors',
 #   'Ethnicity',
 #   'Smoking Status',
 #   'Alcohol Consumption',
 #   'Steroid Use History',
 #   'Genetic Testing'
#]

# Eliminar las columnas innecesarias
#df = df.drop(columns=columnas_a_eliminar)

clases = df['Target'].unique()
num_clases = len(clases)

muestras_por_clase = 10000 // num_clases  


conteos = df['Target'].value_counts()
if any(conteos < muestras_por_clase):
    # Si alguna clase tiene menos muestras de las requeridas, 
    # no podremos hacer la distribución exacta.
    # Podrías ajustar la cantidad total o la lógica.
    raise ValueError("No hay suficientes datos en al menos una clase para alcanzar el equilibrio deseado.")

df_list = []
for c in clases:
    df_clase = df[df['Target'] == c]
    df_clase_sample = df_clase.sample(n=muestras_por_clase, random_state=42)
    df_list.append(df_clase_sample)

df_muestra = pd.concat(df_list, axis=0)

# Guardar el nuevo conjunto de datos
df_muestra.to_csv('data_limpio.csv', index=False)


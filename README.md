# TC3002B Fact-Checking Model
Angel Mauricio Ramírez Herrera | A01710158
## 1. Propósito del Proyecto  

El objetivo de este proyecto es desarrollar un modelo de **Procesamiento de Lenguaje Natural (NLP)** capaz de detectar desinformación en afirmaciones científicas escritas en lenguaje natural.

El sistema busca clasificar automáticamente afirmaciones en tres categorías:

- **Supported** → La afirmación está respaldada por consenso científico.
- **Refuted** → La afirmación contradice evidencia científica establecida.
- **Not Enough Evidence** → No existe evidencia suficiente para confirmar o refutar la afirmación.

Este proyecto tiene como finalidad:

- Contribuir al desarrollo de sistemas automatizados de verificación científica.
- Explorar técnicas de clasificación multiclase en NLP.
- Evaluar distintos enfoques de modelado (TF-IDF + ML, Deep Learning, Transformers).
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import kagglehub
import os
```

```python
def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

def count_empty(df):
    return df.select([
        sum(when(trim(col(c)) == "", 1).otherwise(0)).alias(c)
        for c in df.columns if dict(df.dtypes)[c] == 'string'
    ])
```

# 2. Dataset Utilizado: SCVD-11K  

Se utilizó el **Scientific Claim Verification Dataset (SCVD-11K)**, un dataset estructurado para clasificación de afirmaciones científicas.

## Resumen del Dataset

- 10,000 muestras de entrenamiento  
- 1,000 muestras de prueba  
- Total: 11,000 afirmaciones etiquetadas  
- Sin entradas duplicadas  
- Idioma: Inglés  
- Distribución de clases: Balanceada
```python
# Download latest version
path = kagglehub.dataset_download("sudhanshuyadav09/scientific-fact-check-classification-dataset")

print("Path to dataset files:", path)

size_bytes = get_dir_size(path)
size_mb = size_bytes / (1024 * 1024)

print(f"Tamaño total de los archivos: {size_mb:.2f} MB")
```

```text
Path to dataset files: /kaggle/input/datasets/sudhanshuyadav09/scientific-fact-check-classification-dataset
Tamaño total de los archivos: 0.84 MB
```

```python
files = os.listdir(path)
print(files)
train = [f for f in files if f.endswith('.csv')][1]
train_path = os.path.join(path, train)

test = [f for f in files if f.endswith('.csv')][0]
test_path = os.path.join(path, test)
```

```text
['scientific_claim_test.csv', 'scientific_claim_train.csv']
```

# 3. Estructura y Formato de los Datos  

## Archivos

- `scientific_claim_train.csv` → Datos para entrenamiento (10,000 filas)
- `scientific_claim_test.csv` → Datos para evaluación (1,000 filas)
```python
df_train = pd.read_csv(
    train_path,
    on_bad_lines='skip',
    low_memory=False
)

df_train.shape
```

```text
(10000, 2)
```

```python
df_test = pd.read_csv(
    test_path,
    on_bad_lines='skip',
    low_memory=False
)

df_test.shape
```

```text
(1000, 2)
```

# 4. Unión de Datasets  

Se unificaron los conjuntos de entrenamiento y prueba con el objetivo de:

- Realizar un Análisis Exploratorio de Datos (EDA) completo.
- Analizar distribución global de clases.
- Detectar posibles inconsistencias o anomalías.
- Aplicar tokenización de manera uniforme.
- Garantizar consistencia en el preprocesamiento.

Posteriormente, para el entrenamiento del modelo, se respetó la separación original entre train y test.
```python
df_merged = pd.concat([df_train, df_test], ignore_index=True)
```

```python
df_merged.shape
```

```text
(11000, 2)
```

## Columnas

| Columna     | Tipo     | Descripción |
|------------|----------|-------------|
| statement  | string   | Afirmación científica en lenguaje natural. |
| label      | string   | Etiqueta categórica (Supported, Refuted, Not Enough Evidence). |
```python
df_merged.head()
```

```text
                                           statement                label
0  Dark matter might interact weakly with normal ...  Not Enough Evidence
1             The Earth is flat. (Statement ID 5654)              Refuted
2  Climate change is a hoax created by scientists...              Refuted
3  Some diets might improve long-term health outc...  Not Enough Evidence
4  A new vaccine candidate may reduce infection r...  Not Enough Evidence
```

```python
df_merged.tail()
```

```text
                                               statement                label
10995  Goldfish have a memory span of only three seco...              Refuted
10996  The Earth revolves around the Sun. (Statement ...            Supported
10997  Photosynthesis allows plants to convert sunlig...            Supported
10998  Lightning never strikes the same place twice. ...              Refuted
10999  A new vaccine candidate may reduce infection r...  Not Enough Evidence
```

```python
(df_merged.isnull().sum() / len(df_merged) * 100).round(2).sort_values(ascending = False)
```

```text
statement    0.0
label        0.0
dtype: float64
```

```python
df_merged['label'].value_counts().plot(kind='bar')

plt.title('Distribución de etiquetas')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()
```

![output image 17-0](images/cell-17-0.png)

# 5. Justificación del Procesamiento de Datos  

El procesamiento es una etapa en proyectos de NLP porque los modelos no trabajan directamente con texto crudo, sino con representaciones numéricas.

## 5.1 Lowercasing  

Se convirtió todo el texto a minúsculas para:

- Reducir la dimensionalidad del vocabulario.
- Evitar que el modelo trate como diferentes palabras:
  - `Vaccine`
  - `vaccine`
```python
df_merged['statement_clean'] = (
    df_merged['statement']
    .str.replace(r'\(Statement ID \d+\)$', '', regex=True)
    .str.replace(r'\[.*?\]', '', regex=True)
    .str.replace(r'\n+', ' ', regex=True)
    .str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
    .str.lower()
)
```

```python
df_merged.head(20)
```

```text
                                            statement                label  \
0   Dark matter might interact weakly with normal ...  Not Enough Evidence   
1              The Earth is flat. (Statement ID 5654)              Refuted   
2   Climate change is a hoax created by scientists...              Refuted   
3   Some diets might improve long-term health outc...  Not Enough Evidence   
4   A new vaccine candidate may reduce infection r...  Not Enough Evidence   
5   Gravity causes objects to fall toward the Eart...            Supported   
6   Photosynthesis allows plants to convert sunlig...            Supported   
7   A new particle could exist beyond the Standard...  Not Enough Evidence   
8   Goldfish have a memory span of only three seco...              Refuted   
9   Climate change is a hoax created by scientists...              Refuted   
10  A specific gene might influence intelligence. ...  Not Enough Evidence   
11  Antibiotics cure viral infections. (Statement ...              Refuted   
12  A specific gene might influence intelligence. ...  Not Enough Evidence   
13  A new particle could exist beyond the Standard...  Not Enough Evidence   
14  Goldfish have a memory span of only three seco...              Refuted   
15  Gravity causes objects to fall toward the Eart...            Supported   
16  The Earth revolves around the Sun. (Statement ...            Supported   
17  Goldfish have a memory span of only three seco...              Refuted   
18  Dark matter might interact weakly with normal ...  Not Enough Evidence   
19  Goldfish have a memory span of only three seco...              Refuted   

                                      statement_clean  
0   dark matter might interact weakly with normal ...  
1                                   the earth is flat  
2      climate change is a hoax created by scientists  
3   some diets might improve long term health outc...  
4   a new vaccine candidate may reduce infection r...  
5     gravity causes objects to fall toward the earth  
6   photosynthesis allows plants to convert sunlig...  
7   a new particle could exist beyond the standard...  
8   goldfish have a memory span of only three seconds  
9      climate change is a hoax created by scientists  
10       a specific gene might influence intelligence  
11                  antibiotics cure viral infections  
12       a specific gene might influence intelligence  
13  a new particle could exist beyond the standard...  
14  goldfish have a memory span of only three seconds  
15    gravity causes objects to fall toward the earth  
16                  the earth revolves around the sun  
17  goldfish have a memory span of only three seconds  
18  dark matter might interact weakly with normal ...  
19  goldfish have a memory span of only three seconds  
```


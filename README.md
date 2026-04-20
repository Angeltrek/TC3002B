# TC3002B — Fact-Checking Model

**Angel Mauricio Ramírez Herrera | A01710158**  
🔗 [Kaggle Notebook](https://www.kaggle.com/code/angeltrek/tc3002b)

---

## Objetivo

Implementar y comparar múltiples modelos de clasificación sobre el dataset **SCVD-11K**, desde un baseline simple hasta modelos de Deep Learning, para identificar cuál ofrece el mejor desempeño en la tarea de **Fact-Checking** con tres clases:

- **Supported** → Afirmación respaldada por consenso científico.
- **Refuted** → Afirmación contradice evidencia científica.
- **Not Enough Evidence** → No existe evidencia suficiente.

---

## Dataset — SCVD-11K

| Atributo | Valor |
|----------|-------|
| Fuente | [Kaggle — sudhanshuyadav09](https://www.kaggle.com/datasets/sudhanshuyadav09/scientific-fact-check-classification-dataset) |
| Total de muestras | 11,000 |
| Split original | 10,000 train / 1,000 test |
| Clases | `Supported` · `Refuted` · `Not Enough Evidence` |
| Balance | ~3,667 muestras por clase (dataset balanceado) |
| Tipo de texto | Afirmaciones científicas de una oración |
| Tamaño en disco | 0.84 MB |

---

## Modelos implementados

| # | Modelo | Tipo |
|---|--------|------|
| 1 | Dummy Classifier (Most Frequent) | Baseline |
| 2 | Logistic Regression + TF-IDF | ML Clásico |
| 3 | Naive Bayes + TF-IDF | ML Clásico |
| 4 | SVM Lineal + TF-IDF | ML Clásico |
| 5 | Random Forest + TF-IDF | ML Clásico |
| 6 | Gradient Boosting + TF-IDF | ML Clásico |
| 7 | LSTM con Embedding aprendido | Deep Learning |
| 8 | BiLSTM con Embedding aprendido | Deep Learning |
| 9 | CNN + BiGRU + GloVe | Deep Learning |

---

## Pipeline de limpieza de texto

```python
def clean_text(series):
    return (
        series
        .str.replace(r'\(Statement ID \d+\)$', '', regex=True)
        .str.replace(r'\[.*?\]', '', regex=True)
        .str.replace(r'\n+', ' ', regex=True)
        .str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.lower()
    )
```

Pasos: eliminación de IDs entre paréntesis, eliminación de corchetes, reemplazo de saltos de línea, solo caracteres alfabéticos, normalización de espacios, conversión a minúsculas.

---

## Métricas de evaluación

Siguiendo a Sokolova & Lapalme (2009):

| Métrica | Justificación |
|---------|---------------|
| **Accuracy** | Válida dado que el dataset está balanceado. |
| **F1 Macro** | Penaliza igual el error en todas las clases. Recomendada en NLP multiclase. |
| **F1 Weighted** | Complementa al F1 Macro considerando soporte de cada clase. |

---

## Resultados

| Modelo | Accuracy | F1 Macro | F1 Weighted | Tiempo (s) |
|--------|----------|----------|-------------|------------|
| Logistic Regression + TF-IDF | 1.0000 | 1.0000 | 1.0000 | 0.13 |
| Naive Bayes + TF-IDF | 1.0000 | 1.0000 | 1.0000 | 0.10 |
| SVM Lineal + TF-IDF | 1.0000 | 1.0000 | 1.0000 | 0.34 |
| Random Forest + TF-IDF | 1.0000 | 1.0000 | 1.0000 | 0.71 |
| Gradient Boosting + TF-IDF | 1.0000 | 1.0000 | 1.0000 | 2.78 |
| CNN + BiGRU + GloVe | 1.0000 | 1.0000 | 1.0000 | 134.89 |
| BiLSTM (Embedding aprendido) | 1.0000 | 1.0000 | 1.0000 | 311.26 |
| LSTM (Embedding aprendido) | 0.6660 | 0.5553 | 0.5548 | 363.89 |
| Dummy Classifier | 0.3330 | 0.1665 | 0.1664 | 0.00 |

### Configuración TF-IDF

```python
TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
```

### Configuración Deep Learning

```
MAX_WORDS = 20000  |  MAX_LEN = 50  |  EMBED_DIM = 128
Vocabulario real del dataset: 164 tokens
Loss: sparse_categorical_crossentropy  |  Optimizer: Adam(lr=0.001)
Callbacks: EarlyStopping(patience=5) + ReduceLROnPlateau
```

---

## Análisis de resultados

**Baseline (Dummy Classifier)**  
Accuracy ≈ 33% con tres clases balanceadas, F1 Macro muy bajo porque las otras dos clases tienen F1 = 0. Define el piso mínimo que cualquier modelo real debe superar.

**ML Clásico + TF-IDF**  
Sorprendentemente competitivos para un dataset de 11K muestras cortas. Logistic Regression y SVM Lineal lideran en clasificación de texto corto porque TF-IDF captura palabras clave discriminativas. Naive Bayes es eficiente pero asume independencia entre palabras. Random Forest y Gradient Boosting son más costosos y no siempre superan a los modelos lineales en texto corto.

**Deep Learning**  
LSTM captura dependencias secuenciales, útil para negaciones como no hay evidencia. BiLSTM mejora sobre LSTM al procesar el texto en ambas direcciones. CNN + BiGRU + GloVe integra conocimiento semántico preentrenado (GloVe 6B 200d), detección de patrones locales y contexto secuencial bidireccional.

**Observación sobre el dataset**  
El SCVD-11K contiene afirmaciones cortas (una oración) y un vocabulario real de solo 164 tokens, lo que favorece a los modelos que capturan palabras clave (TF-IDF + modelos lineales). Los modelos con suficiente capacidad memorizan el mapping completo, lo que explica el alto accuracy generalizado.

---

## Cómo ejecutar

El notebook está diseñado para correr en **Kaggle con GPU habilitada**. El dataset se descarga automáticamente:

```python
import kagglehub
path = kagglehub.dataset_download('sudhanshuyadav09/scientific-fact-check-classification-dataset')
```

GloVe (Modelo 9) se descarga desde Stanford en la primera ejecución (~862 MB).

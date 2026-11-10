# Práctica 002 — SVM de regresión (SVR) para predecir ventas

Resumen del proyecto de la práctica de **Aprendizaje Automático**. El objetivo es
desarrollar, optimizar y evaluar un modelo **SVM de regresión (SVR)** sobre el dataset de
ventas de la práctica 001, además de exponer una función de predicción para un sample
nuevo y cerrar con conclusiones y referencias en APA 7.

> **Estado:** en desarrollo. R1 entregó las secciones 1–3 y la v4 aprobada ya está
> promovida en `src/svr_ventas.ipynb`. R2 entregó las secciones 4–6 en
> `src/jose/practica-002-v1/modelado_mlflow.ipynb` (en revisión); el avance de cada rol está en
> [Estado por rol](#estado-por-rol).
> La guía de desarrollo está en [`docs/plan_desarrollo_svr_4_personas.md`](docs/plan_desarrollo_svr_4_personas.md)
> y su versión PDF en [`docs/plan_desarrollo_svr_4_personas.pdf`](docs/plan_desarrollo_svr_4_personas.pdf).

---

## Objetivo

Aplicar SVR al dataset de ventas retail, comparar distintos **kernels** y valores de
**penalización `C`**, optimizar los hiperparámetros con `GridSearchCV`, registrar **cada
entrenamiento como una versión** en el Model Registry de **MLflow** y entregar una función
`predecir_venta()` probada con dos casos distintos.

## Alcance

| Ítem | Incluido |
|---|---|
| 3.1 Desarrollo y optimización del modelo | Sí |
| 3.2 Función de predicción para un sample nuevo | Sí |
| 3.4 Conclusiones y referencias APA 7 | Sí |
| Justificación de predicciones (SHAP, PDP/ICE, permutation importance) | **No** |
| Random Forest y árboles de decisión | **No** |

## Dataset

Se reutiliza el dataset de la práctica 001:

- **Archivo obligatorio:** `datos/ventas_features.csv` (600 filas, 22 columnas).
- **Contenido:** 10 productos × 60 días (enero–marzo de 2022), con variables de calendario,
  lags de ventas, promedio móvil, stock y días para expirar.
- **Variable objetivo:** `cantidad_vendida` (continua, unidades por día).
- **Origen:** dataset construido en la práctica 001 a partir del dataset público de Kaggle
  *Retail Store Inventory Forecasting*.

## Enfoque técnico

| Decisión | Valor |
|---|---|
| Modelo | `SVR` de scikit-learn |
| Kernels | `linear`, `rbf`, `poly`, `sigmoid` |
| Penalización `C` | `0.1`, `1`, `10`, `100` |
| Optimización | `GridSearchCV` 5-fold sobre C, epsilon, gamma y degree |
| Variables | 11 numéricas + `categoria` y `product_id` one-hot (25 columnas; 5 duplicadas por decisión del equipo, ver sección 3.6 del notebook) |
| Escalado | `StandardScaler` en X y en y (`TransformedTargetRegressor`) |
| Split | Temporal 80/20 por fecha |
| Métricas | RMSE (principal), MAE, R², MAPE |
| Tracking | MLflow local con SQLite (`mlflow.db`) y Model Registry |
| Versiones | `svr_ventas` v1–v16 (barrido) + v17 (`GridSearchCV`), alias `champion` |
| Semilla | `random_state=42` |

## Resultados esperados

Valores medidos previamente sobre este dataset, útiles como referencia:

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| Baseline ingenuo (`promedio_movil_7d`) | 41.72 | 31.42 | — |
| SVR linear, C=1, split temporal | 26.27 | 20.12 | 0.768 |
| SVR RBF, C=10, split aleatorio | 22.42 | 16.68 | 0.848 |
| LSTM de la práctica 001 (referencia) | 23.19 | 17.30 | — |

> Estos valores se midieron durante la planificación con configuraciones distintas a la
> final (una fila usa split aleatorio y otra selección de columnas); son una referencia
> orientativa. La tabla consolidada de la sección 8 del notebook, generada desde el
> registry, es la fuente definitiva.

La conclusión esperada es que un SVR simple alcanza un desempeño comparable al LSTM de la
práctica anterior con mucha menor complejidad.

## Entregables previstos

| Archivo | Descripción |
|---|---|
| `src/svr_ventas.ipynb` | Notebook único con datos, modelo, MLflow, evaluación y predicción |
| `src/svr_ventas.html` | Exportación del notebook para entrega |
| `mlflow.db` + `mlartifacts/` | Registry (17 versiones) y artefactos de los runs (locales, ignorados en Git) |
| `resultados/modelos/` | Un `.joblib` por versión, para revisión sin MLflow |
| `resultados/resultados_modelos.csv` | Tabla consolidada de todos los modelos |
| `resultados/tabla_variables.csv` | Variables usadas y excluidas, con el motivo de cada exclusión |
| `resultados/figuras/` | Figuras de EDA, comparación y evaluación |
| `requirements.txt` | Dependencias del entorno con versiones fijas |
| `docs/entorno.md` | Guía de instalación y verificación del entorno |

## Estructura del repositorio

```text
practica-002/
├── datos/
│   ├── ventas_features.csv            # dataset principal
│   ├── sales_history.csv              # ventas crudas (trazabilidad)
│   └── inventory.csv                  # catálogo de productos
├── docs/
│   ├── entorno.md                     # instalación y verificación del entorno
│   ├── guia.md                        # enunciado de la práctica
│   ├── guia.docx                      # enunciado original
│   ├── plan_desarrollo_svr_4_personas.md
│   └── plan_desarrollo_svr_4_personas.pdf
├── src/
│   ├── migue/
│   │   ├── practica-002-v1/           # entrega v1 de R1 (histórico)
│   │   ├── practica-002-v2/           # entrega v2 de R1 (histórico)
│   │   ├── practica-002-v3/           # entrega v3 de R1 (histórico)
│   │   └── practica-002-v4/           # entrega v4 de R1 (aprobada)
│   ├── jose/
│   │   └── practica-002-v1/           # entrega v1 de R2 (modelado_mlflow.ipynb, en revisión)
│   └── svr_ventas.ipynb               # integración final (secciones 1–3 listas)
├── resultados/
│   ├── resultados_modelos.csv         # una fila por versión (a generar, R3)
│   ├── tabla_variables.csv            # variables usadas y excluidas (R1)
│   ├── modelos/                       # un .joblib por versión (a generar, R2)
│   └── figuras/                       # 4 figuras del EDA (R1); luego las de R3
├── mlflow.db                          # registry SQLite (ignorado en Git)
├── mlartifacts/                       # artefactos de los runs (ignorado en Git)
├── .python-version                    # 3.12.7
├── .gitignore
├── .gitattributes
├── datos.sha256                       # huellas de los CSV
├── requirements.txt                   # dependencias con versión fija
└── requirements-lock.txt              # a generar con pip freeze
```

## Equipo y roles

| Rol | Responsabilidad | Carga |
|---|---|---:|
| R1 — Datos y EDA | Dataset, features, split y control de fuga | 8 h |
| R2 — Modelado y MLflow | Pipeline SVR, barrido, GridSearch y tracking | 10 h |
| R3 — Evaluación y predicción | Métricas, figuras, `predecir_venta()` y HTML | 9 h |
| R4 — Conclusiones y documentación | Conclusiones, APA y revisión final | 6 h |

El detalle de tareas, entregables y criterios de aceptación está en la sección 5 del plan.

### Estado por rol

| Rol | Secciones de `src/svr_ventas.ipynb` | Estado |
|---|---|---|
| R1 — Datos y EDA | 1 Configuración, 2 Datos y EDA, 3 Features y split | Listo (v4 aprobada y promovida) |
| R2 — Modelado y MLflow | 4 Pipeline SVR, 5 Barrido kernels × C, 6 Optimización | En revisión (v1 en `src/jose/practica-002-v1/`) |
| R3 — Evaluación y predicción | 7 Evaluación, 8 Resultados consolidados, 9 Función de predicción | Pendiente |
| R4 — Conclusiones y documentación | 0 Portada, 10 Conclusiones, 11 Referencias | Pendiente |

R1 entrega, para que nadie los modifique: `X_train`, `X_test`, `y_train`, `y_test`,
`COLUMNAS` (25 columnas), `df` (limpio, 530 filas), `SEED` y `guardar()`. La revisión de
fuga y los hallazgos para el equipo están en las secciones 3.5 y 3.6 del notebook.

R2 entrega `construir_svr()`, `mejor_modelo` y el modelo registrado `svr_ventas`: 17 runs y 17
versiones (v1–v16 del barrido, v17 de `GridSearchCV` con alias `champion`: `rbf`, C=10,
`epsilon`=0.1, `gamma`=0.01), un `.joblib` por versión en `resultados/modelos/` y la tabla cruda
`tabla_versiones` para la sección 8. La selección por CV frente a test está en la sección 6.3.

## Reproducción

```bash
# 1. Verificar que los datos son los mismos
sha256sum -c datos.sha256

# 2. Crear el entorno (Python 3.12.7)
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check

# 3. Registrar el kernel de Jupyter
python -m ipykernel install --user --name svr-ventas --display-name "Python (svr-ventas)"

# 4. Ejecutar el notebook
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace src/svr_ventas.ipynb

# 5. Exportar a HTML
jupyter nbconvert --to html src/svr_ventas.ipynb

# 6. Ver runs y versiones del modelo
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

El detalle de versiones, verificación y problemas frecuentes está en
[`docs/entorno.md`](docs/entorno.md).

## Documentación

- [`docs/entorno.md`](docs/entorno.md): instalación, versiones y verificación del entorno.
- [`docs/guia.md`](docs/guia.md): enunciado resumido de la práctica.
- [`docs/guia.docx`](docs/guia.docx): guía oficial completa.
- [`docs/plan_desarrollo_svr_4_personas.md`](docs/plan_desarrollo_svr_4_personas.md): plan de
  desarrollo con decisiones, roles, cronograma y guía técnica.
- [`docs/plan_desarrollo_svr_4_personas.pdf`](docs/plan_desarrollo_svr_4_personas.pdf):
  versión PDF del plan (19 páginas, A4).

## Referencias principales

- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.
- Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing, 14*(3), 199–222.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Raschka, S. (2017). *Python machine learning* (2.ª ed.). Packt Publishing.
- Zaharia, M., et al. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin, 41*(4), 39–45.

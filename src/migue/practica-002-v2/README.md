# Práctica 002 — SVR para predecir ventas diarias

Aprendizaje Automático · SVM de regresión (SVR) sobre `ventas_features.csv` de la práctica 001.

## Estructura

```
practica-002/
├── datos/
│   ├── ventas_features.csv       # dataset derivado de la práctica 001 (con rezagos)
│   └── sales_history.csv         # dataset original sin nulos (lo usa el assert de la sección 2.2)
├── resultados/
│   ├── figuras/                  # 4 figuras del EDA (R1); luego las de R3 y R4
│   └── tabla_variables.csv       # variables incluidas / excluidas y su motivo
├── svr_ventas.ipynb              # notebook único (cada rol edita solo su sección)
├── requirements.txt
└── .gitignore
```

## Cómo ejecutar

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace svr_ventas.ipynb
```

## Estado por rol

| Rol | Secciones | Estado |
|---|---|---|
| R1 — Datos y EDA | 1 Configuración, 2 Datos y EDA, 3 Features y split | Listo |
| R2 — Modelado y MLflow | 4 Pipeline, 5 Barrido, 6 Optimización | Pendiente |
| R3 — Evaluación y predicción | 7 Evaluación, 8 Resultados, 9 `predecir_venta()` | Pendiente |
| R4 — Explicabilidad y documentación | 0 Portada, 10 Explicabilidad, 11 Conclusiones, 12 Referencias | Pendiente |

## Contrato de R1 con el resto del equipo

Objetos que no se modifican: `X_train`, `X_test`, `y_train`, `y_test`, `COLUMNAS`, `df` (limpio, 530 filas), `SEED` (42) y `guardar()`.
Las decisiones de limpieza y los hallazgos para el equipo están en las secciones 3.7 y 3.8 del notebook.

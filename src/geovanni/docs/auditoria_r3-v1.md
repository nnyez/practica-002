# Auditoría de la entrega R3 — notebook v1

| Campo | Valor |
|---|---|
| **Revisado** | `src/geovanni/practica-002-v1/` (notebook, HTML, datos, resultados y README) |
| **Alcance** | Secciones 7 a 9 |
| **Fecha** | 11 de noviembre de 2026 |
| **Veredicto** | **APROBADA** — sin P0, P1 ni P2 abiertos; la v1 corregida puede promoverse |

La v1 implementa la evaluación del champion cargado del registry, la tabla consolidada
reconstruida desde las versiones registradas, las tres figuras de evaluación y
`predecir_venta()` con validación de `categoria`/`product_id`. El notebook ejecuta de punta
a punta (31 celdas de código, counts 1–31, 0 errores). Las secciones 1–6 son idénticas a R2
v2, tanto en fuente como en salidas; la única diferencia es la celda de portada. Además de
revisar la entrega, reconstruí el split y recalculé las métricas por fuera del notebook,
cargué los 17 `.joblib` y consulté el registry en `mlflow.db`. Las seis observaciones P2 de
la primera revisión están corregidas y verificadas (tabla al final).

---

## Evidencia verificada

- **Ejecución:** 31 celdas de código con counts 1–31 y 0 errores. Las celdas 16 y 70 no
  tienen salida porque solo definen listas/funciones y ejecutan `assert`; es lo esperado.
- **Secciones 1–6:** fuentes y salidas idénticas a `src/jose/practica-002-v2/` para las 22
  celdas de código; solo cambia la celda de portada (índice 0), que ahora anuncia las
  secciones 7–9.
- **Registry:** `mlflow.db` con 17 versiones (v1–v17) y 17 runs activos, más el run de
  prueba `prueba_configuracion` en estado `deleted`. Los tags `kernel`, `C` y `origen` están
  completos; el alias `champion` apunta a la v17 (`rbf`, C=10, `epsilon`=0.1, `gamma`=0.01),
  la del `GridSearchCV`. `version_inicial` es 0 y las comprobaciones se hacen relativas a
  ella, así que una nueva ejecución sobre el mismo registry agrega 17 versiones y mueve el
  alias.
- **Tabla consolidada:** `resultados/resultados_modelos.csv` tiene 17 filas y 10 columnas,
  ordenado por RMSE. Los 17 `.joblib` de `resultados/modelos/` reproducen las métricas del
  CSV con diferencias menores a `1e-9`; `svr_mejor.joblib` es byte-idéntico al champion de
  R2.
- **Datos:** `sha256sum -c datos.sha256` pasa y los tres CSV tienen el mismo SHA-256 que los
  del raíz.
- **Figuras:** hay 7 (4 del EDA y 3 de evaluación). Las 4 del EDA son byte-idénticas a las
  promovidas en `resultados/figuras/`; `predicho_vs_real.png`, `residuos.png` y
  `comparacion_kernels.png` existen y están embebidas en el HTML.
- **Requisitos:** los pins de `requirements.txt` coinciden con `requirements-lock.txt`
  (`mlflow 3.18.0`, `scikit-learn 1.9.1`, `numpy 2.5.4`, `pandas 3.0.6`, etc.).
- **HTML:** 7 imágenes embebidas, sin traceback, con las secciones 7–9 y la tabla de casos.
- **Prohibiciones:** no hay Random Forest, árboles de decisión ni explicabilidad
  (SHAP/PDP/ICE/permutación/LIME) en el código ni en la narrativa. Las únicas coincidencias
  de texto son subcadenas de `Alimentos` y `shape`.
- **P1 de la revisión anterior:** `predecir_venta()` deriva `categoria` desde el catálogo
  cuando recibe `product_id`, rechaza una categoría que no corresponde, rechaza el caso de
  solo categoría y usa `P0001`/`Alimentos` cuando faltan ambas. Las 6 pruebas de error y las
  2 pruebas positivas pasan.

> El registry no viaja en Git (`mlflow.db` y `mlartifacts/` están ignorados), así que la
> comprobación se apoyó en la base SQLite de la entrega, en las salidas ejecutadas y en la
> reproducción exacta de las métricas.

---

## Reproducción independiente

| Comprobación | Resultado |
|---|---|
| Split reconstruido | 430 train / 100 test; train 8-ene → 19-feb, test 20-feb → 1-mar |
| Champion (v17) | RMSE 26.904020 · MAE 20.182334 · R² 0.756432 · MAPE 12.469459 · RMSE_CV 19.039611 |
| Mejor test del barrido | `linear` C=10: RMSE 26.256174 · MAE 20.079513 · R² 0.768021 · MAPE 12.295356 |
| Baseline ingenuo | RMSE 41.722859 · MAE 31.415714 · R² 0.414222 · MAPE 18.393181 |
| Residuos del champion | media 5.342516 · mediana 5.048794 · desv. 26.501073 · mín −69.864422 · máx 90.121789; 74.0 % dentro de ±RMSE |
| Caso alto | P0008, 26-feb: 332.5 predicho vs 333 real (+0.5; 0.15 %) |
| Caso bajo | P0004, 23-feb: 87.0 predicho vs 100 real (−13.0; 13.00 %) |
| Función vs modelo | `predecir_venta()` coincide con `modelo_final.predict` en los dos casos |
| 17 modelos | todos los `.joblib` reproducen las métricas del CSV con Δ < 1e-9 |

---

## P0

Ninguno.

## P1

Ninguno.

## P2

Ninguno abierto. La primera revisión dejó seis observaciones; R3 las corrigió en `0b964ea`
y esta revisión comprobó cada corrección:

| # | Observación en la v1 recibida | Corrección verificada |
|---|---|---|
| 1 | Salto de 9.3 a 9.5, sin 9.4 | 9.4 «Entrega de R3 a R4» existe; la referencia interna a las pruebas límite sigue apuntando a 9.3 |
| 2 | `tabla_coincide = True` como cuarto criterio del checklist | Eliminado; el checklist recalcula cada booleano en `criterios` y conserva el `assert_frame_equal` |
| 3 | README con el estado y el árbol anteriores a R2 | Estado y tabla por rol con R1–R3 aprobadas; árbol con secciones 1–9, 17 `.joblib` y CSV de R3 |
| 4 | Enlaces del README sin resolver desde la carpeta de entrega | Apuntan a `../../../docs/` e incluyen `../docs/auditoria_r3-v1.md` y `../../jose/docs/auditoria_r2-v2.md` |
| 5 | `kernelspec` `python3` en vez de `svr-ventas` | `kernelspec` = `svr-ventas` |
| 6 | MAPE sin unidad en el README | La tabla dice `MAPE (%)`; la columna del CSV conserva el nombre `MAPE` |

---

## Cierre

- [x] Renumerar 9.5 como 9.4.
- [x] Quitar `tabla_coincide = True` y dejar la comprobación real en `criterios`.
- [x] README de la entrega con el estado y el árbol actualizados a R1–R3 aprobados.
- [x] README de la entrega con enlaces válidos y enlace a `auditoria_r3-v1.md`.
- [x] `kernelspec` del notebook = `svr-ventas` (solo metadatos).
- [x] Unidad de MAPE documentada como porcentaje en el README.

Sin P0, P1 ni P2 abiertos, la v1 puede promoverse a `src/svr_ventas.ipynb`. Las seis
correcciones fueron de numeración, metadatos y texto; no cambian código de modelado ni
resultados, y no requieren volver a entrenar.

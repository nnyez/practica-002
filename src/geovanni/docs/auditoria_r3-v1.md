# Auditoría de la entrega R3 — notebook v1

| Campo | Valor |
|---|---|
| **Revisado** | `src/geovanni/practica-002-v1/` (notebook, HTML, datos, resultados y README) |
| **Alcance** | Secciones 7 a 9 |
| **Fecha** | 11 de noviembre de 2026 |
| **Veredicto** | **APROBADA** — sin P0 ni P1; quedan 6 observaciones P2, ninguna bloquea la promoción |

La v1 implementa la evaluación del champion cargado del registry, la tabla consolidada
reconstruida desde las versiones registradas, las tres figuras de evaluación y
`predecir_venta()` con validación de `categoria`/`product_id`. El notebook ejecuta de punta
a punta (31 celdas de código, counts 1–31, 0 errores). Las secciones 1–6 son idénticas a R2
v2, tanto en fuente como en salidas; la única diferencia es la celda de portada. Además de
revisar la entrega, reconstruí el split y recalculé las métricas por fuera del notebook,
cargué los 17 `.joblib` y consulté el registry en `mlflow.db`.

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

### 1. Salto de numeración de 9.3 a 9.5

**Por qué.** La sección 9.3 es «Pruebas límite» y la siguiente es 9.5 «Entrega de R3 a R4»;
no existe 9.4. Quien lee el índice o busca la subsección faltante no sabe si se perdió un
bloque o si solo se saltó un número. La referencia interna de 9.5 a «las pruebas límite en
9.3» sigue siendo correcta.

**Qué.** Renumerar 9.5 como 9.4, o agregar el 9.4 que se había previsto. Es solo el
encabezado; no hay que reejecutar nada.

### 2. El criterio `tabla_coincide = True` es tautológico

**Por qué.** En la celda 63 se asigna `tabla_coincide = True` y la celda 77 lo usa como
cuarto criterio del checklist. Ese ítem siempre se muestra en ✔ aunque la comparación real
falle; la comprobación efectiva es el `assert_frame_equal` de la celda 63, que sí detiene el
notebook. El checklist da una falsa sensación de verificación independiente.

**Qué.** Eliminar el ítem del checklist y dejar el `assert_frame_equal` como única
comprobación, o calcular el booleano con una comparación real (por ejemplo, `np.allclose`
columna a columna sobre las columnas numéricas y `equals` sobre `kernel`, `C`, `origen` y
`run`).

### 3. El README de la entrega conserva el estado y el árbol anteriores a R2

**Por qué.** El bloque «Estado» y la tabla «Estado por rol» dicen que R2 está «En revisión»,
pero en el raíz ya figura como «Listo (v2 aprobada y promovida)». El árbol sigue diciendo
`src/svr_ventas.ipynb # integración final (secciones 1–3 listas)`, cuando el notebook ya
cubre las secciones 1–9, y describe `resultados_modelos.csv` como de R2. La entrega de R3 se
lee como si el proyecto estuviera en un estado anterior al real.

**Qué.** Actualizar el estado a «R2 v2 aprobada y promovida» y «R3 v1 aprobada y
promovida», y corregir el árbol: `src/svr_ventas.ipynb # integración final (secciones 1–9;
R1–R3)`, `practica-002-v1/ # entrega v1 de R3 (aprobada)`,
`resultados_modelos.csv # una fila por versión, ordenada por RMSE (R3)` y
`modelos/ # 17 .joblib (16 del barrido + champion)`.

### 4. Los enlaces relativos del README no resuelven desde la carpeta de entrega

**Por qué.** El README es una copia del raíz: enlaza `docs/entorno.md`, `docs/guia.md`,
`docs/guia.docx`, `docs/plan_desarrollo_svr_4_personas.md` y su PDF, pero
`src/geovanni/practica-002-v1/` no tiene `docs/`. Tampoco incluye el enlace a la auditoría de
R2 ni a la de R3. Quien recibe la carpeta no llega a la guía, al plan ni al entorno.

**Qué.** En la copia de la entrega, apuntar los enlaces a `../../../docs/...` (o quitarlos) y
agregar `../docs/auditoria_r3-v1.md`. La copia del raíz se actualiza al promover.

### 5. El kernel del notebook no es el documentado

**Por qué.** `docs/entorno.md` y el README indican instalar y seleccionar el kernel
`svr-ventas`; el notebook declara `kernelspec` con `name: python3` y
`display_name: .venv (3.12.7)`. Quien abra la entrega en otra máquina verá un kernel que
puede no existir y el paso de reproducción del README no coincidirá con lo que el archivo
declara. No afecta a los resultados: la celda 1 valida las versiones instaladas.

**Qué.** Dejar `metadata.kernelspec = {"name": "svr-ventas", "display_name": "Python
(svr-ventas)", "language": "python"}`. Es solo metadato; no hay que reejecutar.

### 6. La unidad de MAPE no está escrita en el README

**Por qué.** `evaluar()` multiplica por 100 y el CSV guarda 12.469, pero la tabla de
métricas del README dice solo «MAPE». El notebook ya usa «MAPE (%)» en las tablas de las
secciones 7 y 8, así que la copia del README es la que puede inducir a leerlo como fracción
(1247 %).

**Qué.** Cambiar «MAPE» por «MAPE (%)» en la tabla de enfoque técnico del README. Mantener el
nombre de la columna `MAPE` para no romper el contrato con R2.

---

## Cierre

- [ ] Renumerar 9.5 como 9.4 (o agregar el 9.4 previsto).
- [ ] Reemplazar `tabla_coincide = True` por una comprobación real o quitar el ítem del
  checklist.
- [ ] README de la entrega con el estado y el árbol actualizados a R2/R3 aprobados.
- [ ] README de la entrega con enlaces válidos y enlace a `auditoria_r3-v1.md`.
- [ ] `kernelspec` del notebook = `svr-ventas` (solo metadatos).
- [ ] Unidad de MAPE documentada como porcentaje en el README.

Sin P0 ni P1, la v1 puede promoverse a `src/svr_ventas.ipynb`. Los seis P2 se resuelven sin
volver a entrenar: son numeración, metadatos y texto.

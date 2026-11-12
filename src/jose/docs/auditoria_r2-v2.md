# Auditoría de la entrega R2 — notebook v2

| Campo | Valor |
|---|---|
| **Revisado** | `src/jose/practica-002-v2/` (notebook, HTML, datos, resultados y README) |
| **Alcance** | Secciones 1 a 6 |
| **Fecha** | 11 de noviembre de 2026 |
| **Veredicto** | **APROBADA** — sin P0 ni P1; quedan 4 observaciones P2, ninguna bloquea la promoción |

La v2 corrige los dos puntos de la revisión anterior: el registry ya no se limpia, la
numeración es relativa a `version_inicial` y la tabla se filtra a los runs de la ejecución.
El notebook ejecuta de punta a punta (22 celdas, counts 1–22, 0 errores) y las secciones 1–3
llegan **idénticas** al notebook raíz: la única diferencia en las 30 celdas comunes es el
texto de portada que describe qué contiene el archivo. Las 8 comprobaciones de la sección
6.4 pasan, `version_inicial` es 0 y `champion` apunta a la v17 (`rbf`, C=10, `epsilon`=0.1,
`gamma`=0.01). Además de revisar la entrega, reconstruí el split y recalcule las métricas de
los 17 modelos y el `GridSearchCV` completo por fuera del notebook.

---

## Evidencia verificada

- **Ejecución:** 22 celdas de código con counts 1–22 y 0 errores. La celda 7 no tiene salida
  porque solo define listas y ejecuta `assert`; es lo esperado.
- **Secciones 1–3:** fuentes idénticas a `src/svr_ventas.ipynb`; solo cambia la celda de
  portada (índice 0).
- **Registry:** 8/8 criterios en ✔, 17 run IDs únicos con formato hexadecimal de 32
  caracteres, un `.joblib` por versión y `champion` cargable desde el alias.
- **Tabla consolidada:** 17 filas, columnas del plan más `origen` y `RMSE_CV`, ordenada por
  RMSE; el mejor test del barrido es la v3 (`linear`, C=10, RMSE 26.256) y el `champion`
  elegido por CV es la v17 (RMSE de test 26.904, RMSE de CV 19.040).
- **Modelos:** los 17 `.joblib` cargan y sus hiperparámetros coinciden con la tabla; el
  baseline ingenuo (`promedio_movil_7d`) queda en 41.72 de RMSE, muy por encima de todos los
  SVR lineales.
- **Datos:** `sha256sum -c datos.sha256` pasa; `requirements.txt` es idéntico al del raíz y
  el lock incluye los pins requeridos (`mlflow 3.18.0`, `scikit-learn 1.9.1`, etc.).
- **Figuras:** tres de las cuatro son byte-idénticas a las promovidas. `eda_temporal.png`
  difiere en 57 píxeles de 1.7 millones (0.003 %), confinados a una etiqueta de fecha en la
  parte superior: diferencia de antialiasing, no de datos.
- **Prohibiciones:** no hay Random Forest, árboles de decisión ni explicabilidad
  (SHAP/PDP/ICE/permutación/LIME) en el código ni en la narrativa.
- **HTML:** 22 prompts `In[n]`, sin traceback, con las 4 figuras embebidas y la sección 6.4.

> El registry no viaja en Git (`mlflow.db` y `mlartifacts/` están ignorados), así que la
> comprobación se apoyó en las salidas ejecutadas del notebook y en la reproducción exacta
> de las métricas, no en inspeccionar la base.

---

## Reproducción independiente

| Comprobación | Resultado |
|---|---|
| Split reconstruido | 430 train / 100 test; train 8-ene → 19-feb, test 20-feb → 1-mar |
| Matriz `X` | (530, 25); 5 dummies duplicadas (`P0006`–`P0010` = `categoria_*`) |
| 17 modelos: ΔRMSE, ΔMAE, ΔR², ΔMAPE, ΔRMSE_CV | 0.0 hasta 9 decimales (por debajo de 1e-12) |
| `GridSearchCV` reconstruido | 72 combinaciones; mismo mejor: `rbf`, C=10, `epsilon`=0.1, `gamma`=0.01 |
| `rmse_cv` del `champion` | 19.039611 |
| Test del `champion` | RMSE 26.904020 · MAE 20.182334 · R² 0.756432 · MAPE 12.469459 |
| Predicciones del `champion` vs `svr_mejor.joblib` | idénticas |

---

## P0

Ninguno.

## P1

Ninguno.

## P2

### 1. El kernel del notebook no es el documentado

**Por qué.** `docs/entorno.md` indica instalar y seleccionar el kernel `svr-ventas`; el
notebook declara `kernelspec` con `name: python3` y `display_name: .venv (3.12.7)`. Quien
abra la entrega en otra máquina verá un kernel que puede no existir y el paso de
reproducción del README no coincidirá con lo que el archivo declara. No afecta a los
resultados: la celda 1 valida las versiones instaladas.

**Qué.** Dejar `metadata.kernelspec = {"name": "svr-ventas", "display_name": "Python
(svr-ventas)", "language": "python"}`. Es solo metadato; no hay que re-ejecutar.

### 2. El README de la entrega es la copia del raíz

**Por qué.** Los enlaces `docs/...` no resuelven desde `src/jose/practica-002-v2/` (esa
carpeta no tiene `docs/`) y el bloque Estado dice que R2 "está en revisión" refiriéndose a
este mismo notebook. Quien recibe la carpeta no llega a la guía, al plan ni a la guía de
entorno.

**Qué.** En la copia de la entrega, apuntar los enlaces a `../../../docs/...` (o quitarlos)
y cambiar el estado a "v2 aprobada, pendiente de promoción". La copia del raíz se actualiza
al promover.

### 3. El árbol del README quedó desactualizado

**Por qué.** Dice `modelos/ # ... (a generar, R2)` y `src/svr_ventas.ipynb # ...
(secciones 1–3 listas)`, pero la entrega trae los 17 `.joblib` y el notebook ya cubre las
secciones 1–6. El árbol es la primera referencia de estructura del repositorio.

**Qué.** Corregir al promover: `modelos/ # 17 .joblib (16 del barrido + champion)`,
`svr_ventas.ipynb # integración final (secciones 1–6; R1–R2)` y
`practica-002-v2/ # entrega v2 de R2 (aprobada)`.

### 4. La unidad de MAPE no está escrita

**Por qué.** `evaluar()` multiplica por 100 y el CSV guarda 12.469, pero ni el notebook ni
el README dicen que es porcentaje. R3 tomará esa columna para la tabla de la sección 8 y
puede leerla como fracción (1247 %) o rotularla mal.

**Qué.** Documentar "MAPE (%)" en el markdown de las secciones 5 y 6 y en la lista de
métricas del README. Mantener el nombre de la columna `MAPE` para no romper el contrato con
R3.

---

## Cierre

- [ ] `kernelspec` del notebook = `svr-ventas` (solo metadatos).
- [ ] README de la entrega con enlaces válidos y estado "aprobada, pendiente de promoción".
- [ ] Árbol del README corregido al promover.
- [ ] Unidad de MAPE documentada como porcentaje.

Sin P0 ni P1, la v2 puede promoverse a `src/svr_ventas.ipynb`. Los cuatro P2 se resuelven
sin volver a entrenar: dos son metadatos o texto y dos se corrigen en la copia raíz durante
la promoción.

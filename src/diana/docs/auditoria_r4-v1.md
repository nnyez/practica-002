# Auditoría de la entrega R4 — notebook v1

| Campo | Valor |
|---|---|
| **Revisado** | `src/diana/practica-002-v1/svr_ventas.ipynb` |
| **Alcance** | Portada, secciones 10–11, cifras de las conclusiones y regresión sobre las secciones 1–9 |
| **Fecha** | 11 de noviembre de 2026 |
| **Veredicto** | **APROBADA CON CAMBIOS** — sin P0; 2 P1 y 5 P2 que corregir antes de promover |

La entrega agrega la portada, las conclusiones y las referencias, y esas tres piezas son
recuperables con correcciones puntuales. El notebook ejecuta sin errores (31 celdas de
código, counts 1–31, 0 errores) y el código y las salidas son idénticos a los del notebook
raíz. Sin embargo, las conclusiones no cumplen el criterio 3.4 completo (faltan R², el valor
de C y la comparación con la práctica 001), las referencias no se citan en el texto y la
reescritura de las secciones 1–9, que es parte de su encargo, pierde tres justificaciones
técnicas que conviene reponer. Con los dos P1 corregidos y esas justificaciones repuestas,
la entrega completa puede promoverse.

---

## Evidencia verificada

- **Ejecución:** 79 celdas (31 de código y 48 Markdown), counts 1–31 y 0 errores. El código
  y las 31 salidas son **byte a byte idénticos** a `src/svr_ventas.ipynb`, incluidos los
  `run_id` y los números de versión 1–17.
- **Portada:** título, universidad, los cuatro integrantes, objetivo y dataset. Las cifras
  de la portada son correctas: 10 productos, 60 días de historial (1-ene a 1-mar-2022) y 530
  filas después de `dropna` (600 − 70).
- **Secciones 10–11:** existen y están al final del notebook (celdas 77 y 78). La sección 10
  tiene cinco viñetas y un párrafo de cierre; la sección 11 tiene seis referencias.
- **Cifras:** todas las que aparecen en las conclusiones y en los párrafos conservados
  coinciden con `resultados/resultados_modelos.csv` y con el dataset, salvo las dos
  imprecisiones del P2 1.
- **Prohibiciones:** no hay Random Forest, árboles de decisión ni explicabilidad
  (SHAP/PDP/ICE/permutación/LIME) en el código ni en la narrativa.
- **Kernel:** `metadata.kernelspec` es `svr-ventas`; coincide con `docs/entorno.md`.

> Las 31 salidas idénticas a la raíz indican que la entrega **reutiliza** las salidas de R3:
> una ejecución nueva sobre el mismo `mlflow.db` habría registrado las versiones 18–34 y
> generado otros `run_id`. Las secciones 10–11 son Markdown y no cambian los resultados, pero
> la entrega no acredita una ejecución propia del notebook.

---

## Cifras de la entrega frente a la medición

| Afirmación | Medición | Resultado |
|---|---|---|
| 530 registros de modelado | 530 (600 − 70) | ✔ |
| Baseline 41.723 de RMSE | 41.722859 | ✔ |
| Champion 26.904 de RMSE en test | 26.904020 | ✔ |
| Mejora del 35,5 % sobre el baseline | 35,51 % | ✔ |
| Champion 19.040 de RMSE en CV | 19.039611 | ✔ |
| Mejor test `linear` C=10 con 26.256 | 26.256174 | ✔ |
| `rbf` C=10 con 30.15 de RMSE | 30.149849 | ✔ |
| `rbf` y `linear` mejores que `poly` y `sigmoid` | 26.26–30.15 frente a 31.02–10 485.82 | ✔ |
| «Poca representación de días feriados» | 0 feriados en train; 2 en test | ✘ |
| «`linear` y `rbf` con C=1 y C=10 entre 26 y 30» | `rbf` C=10 = 30.15 | ✘ |

---

## Regresión sobre las secciones 1–9

De las 49 celdas Markdown del raíz, solo cuatro se conservaron textualmente (las
interpretaciones de 2.5, 3.6, 5.3 y 6.3). Las otras 45 se reescribieron y se eliminaron tres
celdas:

- **Traspaso de R2 a R3 y R4** (celda 50 del raíz): lista el `champion`, la ruta del CSV, las
  métricas que R4 debía usar en las conclusiones y las limitaciones a reportar.
- **9.4 Entrega de R3 a R4** (celda 78 del raíz): métricas de test del champion, mejor test
  del barrido, baseline, LSTM de la práctica 001, figuras, casos y limitaciones. Era la fuente
  directa de las conclusiones.
- **Nota de cierre** (celda 79 del raíz).

La sección 9 ya no salta de 9.3 a 9.5: al eliminar 9.4 la numeración termina en 9.3 y sigue
la sección 10, que es correcto. Lo que se pierde es el contenido del traspaso, no la
numeración. La verificación de 9.3 también se redujo: ya no enumera las garantías del
contrato (mensaje por campo numérico, lista de valores válidos para categoría/producto
desconocidos, fallo de la combinación imposible, categoría derivada del catálogo y ausencia
de imputación).

La reescritura no introduce código nuevo ni cambia resultados. Varias celdas mejoran la
redacción y quitar los traspasos internos en la versión final es razonable; lo que conviene
reponer es el detalle técnico que se perdió (P2 3).

---

## P0

Ninguno.

## P1

### 1. Las conclusiones no cumplen el criterio 3.4

**Por qué.** El plan exige que las conclusiones citen RMSE, R², kernel y C, y la rúbrica
pide «comparación con la práctica 001». La sección 10 da el RMSE del champion (26.904) y su
CV (19.040), pero **no da el R²** (0.756), **no dice C=10** y **no menciona el LSTM** (23.19),
que sí aparece en la tabla de la sección 7. Tampoco compara el R² del champion con el 0.414
del baseline, que es la otra mitad de la mejora. Con eso, el criterio de aceptación «las
conclusiones citan números concretos (RMSE, R², kernel, C)» no se cumple y la rúbrica de 3.4
queda incompleta.

**Qué.** Agregar a la viñeta del champion: `R²` 0.756, `C=10`, `epsilon`=0.1 y `gamma`=0.01.
Agregar una viñeta que compare con el baseline (RMSE 41.723 → 26.904; R² 0.414 → 0.756) y con
el LSTM de la práctica 001 (RMSE 23.19, sin R², MAPE ni CV), aclarando que la comparación es
orientativa porque el LSTM se evaluó con otro protocolo.

### 2. Las referencias no se citan y el formato APA está incompleto

**Por qué.** Los apellidos de las seis referencias aparecen una sola vez en todo el notebook,
y es en la propia lista: no hay ninguna cita en el texto. El criterio 3.4 exige «referencias en
APA 7, citadas en el texto», y la sección 10 discute SVR, `GridSearchCV` y MLflow sin citar a
Cortes y Vapnik, Smola y Schölkopf, Pedregosa et al. ni MLflow. Además:

- `Machine Learning, 20` y `Statistics and Computing, 14` omiten el número de fascículo, que
  en ambos casos es el 3.
- Faltan las cursivas de nombre de revista y volumen (`*Machine Learning, 20*(3)`).
- Las dos entradas de MLflow comparten autor y año, así que APA 7 exige `(s. f.-a)` y
  `(s. f.-b)`, ordenadas por título (Model Registry antes que Tracking), y los títulos de las
  páginas van en cursiva.
- La página de `GridSearchCV` necesita el título en cursiva.

**Qué.** Citar en el texto al menos una vez cada referencia —por ejemplo «(Cortes & Vapnik,
1995)» al presentar SVR, «(Smola & Schölkopf, 2004)» al explicar `epsilon`/`C`,
«(Pedregosa et al., 2011)» en el pipeline, «(MLflow, s. f.-a)» en el registry y «(MLflow,
s. f.-b)» en el tracking— y corregir los seis registros con el formato de arriba.

## P2

### 1. Dos cifras imprecisas

**Por qué.** «Poca representación de días feriados» sugiere que hay algunos en train, pero no
hay ninguno: el 1-ene se pierde con `dropna` y el 28-feb y el 1-mar caen en test (sección
3.6). «`linear` y `rbf` con C=1 y C=10 se mantienen entre 26 y 30» excluye a `rbf` C=10, que
tiene 30.15. Las dos frases están en celdas que se conservaron del raíz (5.3 y 8.2) y en la
conclusión nueva.

**Qué.** Escribir «el modelo no ve ningún feriado en entrenamiento; los dos feriados del test
concentran los residuos más altos» y «`linear` y `rbf` con C=1 y C=10 quedan entre 26.26 y
30.15».

### 2. Jerarquía de encabezados y negritas

**Por qué.** El título usa `#` y las secciones pasan a `###`, sin ningún `##`; las
subsecciones usan `####`. Además, todos los encabezados van en negritas (`### **1.
Configuración**`), incluidas las de la portada. En Jupyter esto salta un nivel y el índice
del notebook queda plano; el raíz usa `##` para secciones y `###` para subsecciones.

**Qué.** Dejar `## N. Sección`, `### N.M Subsección` y `# Título`, sin negritas dentro del
encabezado.

### 3. La edición final pierde tres justificaciones

**Por qué.** El equipo le encargó a R4 los toques finales del informe, así que la nueva
redacción es válida; el problema es que al condensar se perdieron argumentos que el informe
necesita:

- **3.1:** ya no dice que entran 11 numéricas y dos categóricas en one-hot, ni por qué se
  conserva `categoria` aunque sea función de `product_id`. 3.6 mantiene el diagnóstico, pero
  la sección que decide las variables se queda sin el porqué.
- **2.2:** ya no explica por qué no se imputan los nulos («rellenar inventaría ventas que no
  ocurrieron») ni da los conteos (90 celdas en 70 filas).
- **6.2:** ya no justifica que `champion` se elige por CV y no por test, que es el argumento
  que sostienen 6.3 y las conclusiones.

También se pierden detalles menores en §1 (el alcance de `raiz_proyecto()`) y en 9.3 (la
lista de garantías del contrato).

**Qué.** Reponer esas tres justificaciones y los dos detalles; el resto del recorte puede
quedar.

### 4. Errores de redacción y marcas sueltas

**Por qué.** La celda 4 dice «Con la fecha como fecha real y comprobamos con `assert` que…»,
frase sin sujeto ni concordancia. La celda 20 tiene un `- - - - -` en medio del párrafo y un
salto de línea que corta la idea. La celda 10 tiene un espacio doble («venden  más»). En la
conclusión, `**modelo seleccionado (champion**)` deja el paréntesis fuera de las negritas y
`predecir_venta()` va sin backticks.

**Qué.** Corregir las cuatro frases; no tocar el código.

### 5. La ejecución no queda acreditada

**Por qué.** Las 31 salidas son idénticas a las de R3, con los mismos `run_id` y las versiones
1–17. Una ejecución nueva sobre el mismo `mlflow.db` habría creado las versiones 18–34 y otros
`run_id`; sobre una base limpia, los `run_id` seguirían siendo distintos. El archivo es
válido y ejecutable, pero no demuestra la ejecución de punta a punta que pide el plan.

**Qué.** Re-ejecutar el notebook completo sobre la base que corresponda, o dejar constancia
de que las salidas se heredan de R3.

---

## Cierre

- [ ] Sección 10 con RMSE, R², kernel, C y RMSE de CV del champion.
- [ ] Sección 10 con la comparación contra el baseline y contra el LSTM (23.19), con la
      salvedad de protocolo.
- [ ] Referencias citadas en el texto y corregidas en APA 7 (fascículos, cursivas,
      `s. f.-a`/`s. f.-b`).
- [ ] Detalle técnico perdido en 2.2, 3.1, 6.2, §1 y 9.3 repuesto.
- [ ] «Poca representación de feriados» y «entre 26 y 30» corregidos.
- [ ] Encabezados en `##`/`###`, sin negritas.
- [ ] Frases de las celdas 4, 20 y 77 corregidas; separador suelto eliminado.
- [ ] Ejecución demostrada.

Con los P1 resueltos y el detalle repuesto, la entrega completa puede promoverse a
`src/svr_ventas.ipynb` sin volver a entrenar.

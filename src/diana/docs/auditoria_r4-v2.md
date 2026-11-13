# Auditoría de la entrega R4 — notebook v2

| Campo | Valor |
|---|---|
| **Revisado** | `src/diana/practica-002-v2/svr_ventas.ipynb` |
| **Alcance** | Portada, secciones 10–11, cifras de las conclusiones y regresión sobre las secciones 1–9 |
| **Fecha** | 12 de noviembre de 2026 |
| **Veredicto** | **APROBADA CON CAMBIOS** — sin P0; 1 P1 y 3 P2 que corregir antes de promover |

La v2 cierra los dos P1 de la revisión anterior: las conclusiones ya citan R², `C`, `epsilon`,
`gamma`, el baseline y el LSTM, y las seis referencias están citadas en el texto con fascículos,
cursivas y `s. f.-a`/`s. f.-b`. También repone las justificaciones de 2.2, 3.1 y 6.2 y corrige
las dos cifras imprecisas. Queda un error en la referencia de Pedregosa (perdió la inicial) y
tres P2, todos de edición: encabezados aún en negritas, cinco erratas en el texto y el alcance
de `raiz_proyecto()` sin reponer.

---

## Evidencia verificada

- **Ejecución:** 79 celdas (31 de código y 48 Markdown), counts 1–31 y 0 errores. El código y
  las 31 salidas son **byte a byte idénticos** a `src/svr_ventas.ipynb` y a la v1, que es lo
  esperado al ser el mismo código, el mismo dataset y la misma semilla.
- **Kernel:** `metadata.kernelspec` es `svr-ventas`.
- **Cifras:** todas las de las conclusiones coinciden con `resultados/resultados_modelos.csv`
  (tabla siguiente).
- **Prohibiciones:** no hay Random Forest, árboles de decisión ni explicabilidad
  (SHAP/PDP/ICE/permutación/LIME) en el código ni en la narrativa.
- **Citas:** los seis apellidos aparecen al menos una vez fuera de la lista de referencias:
  Cortes y Vapnik (1995) y Smola y Schölkopf (2004) en 4.2 y 5.2; Pedregosa et al. (2011) en
  4.2; scikit-learn developers (s. f.) en 6; MLflow `s. f.-a` en 5.1 y 6.2; `s. f.-b` en 5 y 5.1.

| Afirmación de la v2 | Medición | Resultado |
|---|---|---|
| Champion: RMSE 26.904, MAE 20.182, R² 0.756, MAPE 12.469 % | 26.904020 / 20.182334 / 0.756432 / 12.469459 | ✔ |
| Champion: RBF, C=10, ε=0.1, γ=0.01, versión 17 | versión 17, `rbf`, 10.0 | ✔ |
| RMSE de CV del champion 19.040 | 19.039611 | ✔ |
| Baseline 41.723 y R² 0.414 | 41.722859 / 0.414 | ✔ |
| Mejora del 35.5 % sobre el baseline | 35,52 % | ✔ |
| `linear` C=10: RMSE 26.256 y R² 0.768 | 26.256174 / 0.768021 | ✔ |
| LSTM de la práctica 001: RMSE 23.19 | 23.19 | ✔ |
| 430 filas (43 días) de train y 100 (10 días) de test | 430 / 100 | ✔ |
| Sin feriados en train; dos en test | 0 / 2 | ✔ |

---

## Estado de los puntos de la auditoría v1

| Punto de la v1 | Estado en la v2 |
|---|---|
| P1 1 — conclusiones sin R², `C` ni LSTM | **Cerrado** |
| P1 2 — referencias sin citas y APA incompleto | **Cerrado**, salvo la errata de Pedregosa |
| P2 1 — «poca representación de feriados» y «entre 26 y 30» | **Cerrado** |
| P2 2 — jerarquía de encabezados y negritas | **Parcial**: se cambió `###` a `##`, pero siguen las negritas y seis subsecciones en `####` |
| P2 3 — justificaciones perdidas en 2.2, 3.1 y 6.2 | **Cerrado**, salvo el alcance de `raiz_proyecto()` |
| P2 4 — errores de redacción y marcas sueltas | **Parcial**: quedan cinco erratas |
| P2 5 — ejecución no acreditada | **Retirado**: las salidas idénticas son lo esperado con el mismo código, dataset y semilla |

---

## P0

Ninguno.

## P1

### 1. La referencia de Pedregosa perdió la inicial

**Por qué.** La entrada dice `edregosa, F., Varoquaux, G., …` y en el texto se cita
«(Pedregosa et al., 2011)». Un apellido mal escrito en la lista de referencias incumple el
criterio de APA 7 y contradice la cita del cuerpo.

**Qué.** Cambiar `edregosa` por `Pedregosa` en la referencia de scikit-learn (celda 78). Es la
única corrección bloqueante.

## P2

### 1. Los encabezados siguen en negritas y con niveles mezclados

**Por qué.** Se aplicó el cambio de `###` a `##` en las secciones, pero todos los encabezados
conservan las negritas (`## **1. Configuración**`, `# **Práctica 002…**`), la sección 11 quedó
en `###` y seis subsecciones siguen en `####` (2.5, 3.1, 3.4, 3.6, 6.4 y 7.1). En Jupyter el
índice sigue plano y la jerarquía es inconsistente.

**Qué.** Dejar `# Título`, `## N. Sección`, `### N.M Subsección`, sin negritas dentro del
encabezado. En la portada, usar `##` para «Universidad» y `###` para «Integrantes»,
«Objetivo» y «Dataset».

### 2. Cinco erratas en el texto

**Por qué.** Son errores de edición que se ven en el PDF final:

- celda 18: «historial suficiente.De esta forma» (falta el espacio).
- celda 20: queda un `=` suelto al final de «9 dummies de `product_id`=».
- celda 22: «-Los registros» (falta el espacio tras el guion).
- celda 46: «su diferencia..» (punto doble).
- celda 77: «dando asi» (falta la tilde).

**Qué.** Corregir las cinco frases; no tocar el código.

### 3. El alcance de `raiz_proyecto()` sigue sin explicarse

**Por qué.** La v1 explicaba que la función localiza la raíz del proyecto desde el directorio de
trabajo, para que el notebook corra aunque se abra desde otra carpeta. La v2 solo dice «se
configuran las rutas del proyecto», que es justamente el detalle que faltaba.

**Qué.** Agregar una línea en §1 con ese alcance. Es opcional si se prefiere dejar la sección
como está, pero completa el detalle que pedía la v1.

---

## Cierre

- [x] Sección 10 con RMSE, R², kernel, C y RMSE de CV del champion.
- [x] Sección 10 con la comparación contra el baseline y contra el LSTM (23.19), con la salvedad
      de protocolo.
- [x] Referencias citadas en el texto y corregidas en APA 7 (fascículos, cursivas,
      `s. f.-a`/`s. f.-b`).
- [ ] Referencia de Pedregosa con la inicial correcta.
- [x] Detalle técnico repuesto en 2.2, 3.1 y 6.2.
- [ ] Alcance de `raiz_proyecto()` repuesto en §1.
- [x] «Poca representación de feriados» y «entre 26 y 30» corregidos.
- [ ] Encabezados en `##`/`###`, sin negritas.
- [ ] Erratas de las celdas 18, 20, 22, 46 y 77 corregidas.

Con la errata de Pedregosa corregida y los tres P2 aplicados, la entrega completa puede
promoverse a `src/svr_ventas.ipynb`.

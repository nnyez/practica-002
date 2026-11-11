# Entorno de desarrollo — Práctica 002

Guía para que los cuatro integrantes trabajen con exactamente las mismas versiones, los
mismos datos y las mismas convenciones. Si algo no coincide con esta tabla, se corrige
antes de escribir código.

| Campo                       | Valor                                                    |
| --------------------------- | -------------------------------------------------------- |
| **Proyecto**          | Práctica 002 — SVR de ventas                           |
| **Python**            | 3.12.7                                                   |
| **Entorno**           | `venv` local en `.venv/`                             |
| **Kernel de Jupyter** | `svr-ventas`                                           |
| **Datos**             | `datos/ventas_features.csv` (hash en `datos.sha256`) |
| **Tracking**          | MLflow local con SQLite (`mlflow.db`) y Model Registry |

---

## 1. Versiones fijadas

| Componente   | Versión         | Uso                                  |
| ------------ | ---------------- | ------------------------------------ |
| Python       | **3.12.7** | Intérprete común                   |
| numpy        | 2.5.4            | Álgebra                             |
| pandas       | 3.0.6            | Carga y transformación de datos     |
| scikit-learn | 1.9.1            | `SVR`, `GridSearchCV`, métricas |
| joblib       | 1.6.0            | Guardado de modelos `.joblib`      |
| matplotlib   | 3.11.2           | Figuras                              |
| seaborn      | 0.13.2           | Figuras                              |
| mlflow       | 3.18.0           | Registro de experimentos y versiones |
| jupyterlab   | 4.6.6            | Notebook                             |
| nbconvert    | 7.17.2           | Exportación HTML                    |
| ipykernel    | 7.4.0            | Kernel                               |

> `shap` **no** se instala: la justificación de predicciones quedó fuera del alcance de la
> práctica. Tampoco se usan Random Forest ni árboles de decisión.

El notebook de R1 se ejecutó originalmente con Python 3.13.18, pandas 3.0.5 y numpy 2.5.3;
desde la entrega v4 ya corre con las versiones de esta tabla, y la celda de configuración
lo comprueba antes de continuar. No hay entorno alternativo: todos usan estas versiones.

---

## 2. Archivos de entorno

| Archivo                   | Para qué                                                                |
| ------------------------- | ------------------------------------------------------------------------ |
| `requirements.txt`      | Dependencias directas con versión exacta                                |
| `requirements-lock.txt` | Dependencias transitivas congeladas (se genera, opcional)                |
| `.python-version`       | Versión de Python para `pyenv`                                        |
| `datos.sha256`          | Huellas de los tres CSV                                                  |
| `.gitignore`            | Evita commitear `.venv/`, `mlflow.db`, `mlartifacts/` y temporales |
| `.gitattributes`        | Fuerza LF y marca los PDF como binarios                                  |

### `requirements.txt`

```txt
# Práctica 002 — SVR de ventas
# Python 3.12.7 (ver .python-version)
# Instalar con: python -m pip install -r requirements.txt

numpy==2.5.4
pandas==3.0.6
scikit-learn==1.9.1
joblib==1.6.0
matplotlib==3.11.2
seaborn==0.13.2

# Tracking de experimentos (obligatorio en la guía)
mlflow==3.18.0

# Notebook y exportación
jupyterlab==4.6.6
nbconvert==7.17.2
ipykernel==7.4.0
```

MLflow instala sus propias dependencias (`pyarrow`, `flask`, `alembic`, etc.); no hace
falta listarlas.

### `requirements-lock.txt` (opcional, recomendado)

Fija también las dependencias transitivas:

```bash
pip freeze > requirements-lock.txt
```

Para reproducir exactamente:

```bash
pip install -r requirements-lock.txt
```

Reglas:

- El lock se regenera solo cuando cambia `requirements.txt` y se avisa al equipo.
- Si alguien usa Windows y el lock se generó en Linux, instala `requirements.txt` y
  reporta diferencias.
- Se commitean los dos archivos.

### `.python-version`

```txt
3.12.7
```

Con `pyenv`, esto selecciona 3.12.7 automáticamente dentro del repositorio.

---

## 3. Instalación

### Linux / macOS

```bash
cd /ruta/a/practica-002
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name svr-ventas --display-name "Python (svr-ventas)"
python -m pip check
```

### Windows PowerShell

```powershell
cd C:\ruta\a\practica-002
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name svr-ventas --display-name "Python (svr-ventas)"
python -m pip check
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Regla: siempre `python -m pip`, nunca `pip` a secas, para no instalar en el Python del
sistema.

---

## 4. Verificación obligatoria

Cada integrante ejecuta esto antes de tocar el notebook:

```bash
python - <<'PY'
import sys, importlib.metadata as md
print("Python", sys.version.split()[0])
for p in ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib",
          "seaborn", "mlflow", "jupyterlab", "nbconvert", "ipykernel"]:
    print(f"{p:12s}", md.version(p))
PY
```

Salida esperada:

```text
Python 3.12.7
numpy        2.5.4
pandas       3.0.6
scikit-learn 1.9.1
joblib       1.6.0
matplotlib   3.11.2
seaborn      0.13.2
mlflow       3.18.0
jupyterlab   4.6.6
nbconvert    7.17.2
ipykernel    7.4.0
```

Y:

```bash
python -m pip check
```

Debe responder:

```text
No broken requirements found.
```

---

## 5. Datos idénticos

Los tres CSV deben ser exactamente los mismos. Desde la raíz del repositorio:

```bash
sha256sum -c datos.sha256
```

Salida esperada:

```text
datos/ventas_features.csv: OK
datos/sales_history.csv: OK
datos/inventory.csv: OK
```

En macOS: `shasum -a 256 -c datos.sha256`.
En Windows: `certutil -hashfile datos\ventas_features.csv SHA256` y comparar a mano.

---

## 6. Jupyter

- Abrir JupyterLab desde el entorno activado: `jupyter lab`.
- Seleccionar el kernel **"Python (svr-ventas)"**, no "Python 3" del sistema.
- Ejecutar el notebook desde la raíz del repositorio: las rutas de `datos/` y
  `resultados/` son relativas a la raíz.

Ejecución sin interfaz:

```bash
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace svr_ventas.ipynb
jupyter nbconvert --to html svr_ventas.ipynb
```

---

## 7. MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- Interfaz: `http://127.0.0.1:5000`.
- Experimento: `svr_ventas`.
- Tracking URI: `sqlite:///mlflow.db`. **No** `file:./mlruns`: el Model Registry exige un
  backend SQL y con `file:` falla.
- Un *run* por combinación de kernel y `C`, y **una versión** del modelo registrado
  `svr_ventas` por run. La versión 17 es la del `GridSearchCV` y lleva el alias
  `champion`.
- Artefactos en `mlartifacts/` (por defecto con backend SQL).
- `mlflow.db` y `mlartifacts/` no se commitean: se regeneran ejecutando el notebook. Los
  modelos para revisión se guardan como `.joblib` en `resultados/modelos/`.

Cargar una versión:

```python
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")
modelo = mlflow.sklearn.load_model("models:/svr_ventas@champion")
```

Si el puerto está ocupado: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.

---

## 8. Git

```bash
git status
```

No deben aparecer `.venv/`, `mlflow.db`, `mlartifacts/`, `__pycache__/` ni
`.ipynb_checkpoints/`.

El archivo `.~lock.guia.docx#` es un temporal de LibreOffice: ya está fuera del índice y
`.gitignore` lo cubre con `.~lock.*#`, así que no debe reaparecer en `git status`.

`.gitattributes` fuerza LF en los `.ipynb`, lo que reduce conflictos al mergear celdas.

---

## 9. Reglas del equipo

1. Nadie instala paquetes sin versión: se agrega con `==` a `requirements.txt` y se
   regenera el lock.
2. `SEED = 42` y `random_state=42` en todo lo aleatorio (split, estimadores,
   `GridSearchCV`).
3. No se usa `shap`, Random Forest ni árboles de decisión.
4. Los notebooks se ejecutan de inicio a fin antes de cada commit, sin warnings.
5. UTF-8 y saltos de línea LF.
6. Si cambia una versión, se avisa al equipo y se actualiza `requirements-lock.txt`.
7. Cada integrante trabaja en `src/<nombre>/`; la integración final va a `src/`.

---

## 10. Problemas frecuentes

| Síntoma                                                      | Causa                                       | Solución                                                                                             |
| ------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: pandas`                               | El notebook usa el kernel del sistema       | Seleccionar "Python (svr-ventas)"                                                                     |
| `AssertionError: No se encontró datos/ventas_features.csv` | El notebook se abrió fuera del repositorio | Ejecutarlo dentro del repo: `raiz_proyecto()` lo ubica desde la raíz, `src/` o `src/<nombre>/` |
| `pip` instala en otra ruta                                  | Se usó `pip` sin `python -m`           | `python -m pip install ...` dentro del venv                                                         |
| `mlflow: command not found`                                 | Venv no activado                            | `source .venv/bin/activate`                                                                         |
| `Address already in use` en MLflow                          | Puerto 5000 ocupado                         | `--port 5001`                                                                                       |
| `Model Registry features are not supported`                 | Tracking URI `file:./mlruns`              | Usar `sqlite:///mlflow.db`                                                                          |
| Gráficas no aparecen al exportar                             | Backend interactivo                         | `MPLBACKEND=Agg`                                                                                    |
| El lock falla en Windows                                      | Se generó en Linux                         | Instalar `requirements.txt` y reportar                                                              |

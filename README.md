
# Predicción de Ventas con Modelos de Regresión

Este proyecto utiliza tres enfoques diferentes para predecir las ventas de un producto con base en varias características como el precio, la publicidad y el stock disponible. Los modelos utilizados son **Random Forest**, **XGBoost** y **Stacking**. A continuación, se describe el propósito del código y los pasos para instalarlo y ejecutarlo.

## Propósito del Código

El propósito de este código es predecir las ventas de un producto utilizando técnicas de regresión. Se generan características a partir de datos simulados, luego se entrenan y evalúan tres modelos diferentes:

- **Random Forest Regressor**: Un modelo de árboles de decisión que utiliza un conjunto de árboles para hacer predicciones.
- **XGBoost Regressor**: Un algoritmo de boosting de gradiente que optimiza el rendimiento a través de la combinación de varios árboles de decisión.
- **Stacking Regressor**: Un modelo que combina los predictores de Random Forest y XGBoost con un meta-modelo para mejorar las predicciones.

El código incluye:
1. La creación y preprocesamiento de datos.
2. Entrenamiento y evaluación de los modelos de regresión.
3. Optimización de hiperparámetros para Random Forest usando **GridSearchCV**.
4. Evaluación de los modelos utilizando **MAE** (Mean Absolute Error) y **RMSE** (Root Mean Squared Error).
5. Visualización de los resultados de las predicciones.

## Requisitos

Para ejecutar este código, necesitas tener Python 3.12 y las siguientes dependencias. Estas se pueden instalar mediante el archivo `requirements.txt`.

### Crear un entorno virtual

1. **Instalar `virtualenv` (si no lo tienes instalado)**:

    ```bash
    pip install virtualenv
    ```

2. **Crear un entorno virtual con Python 3.12**:

    ```bash
    virtualenv venv --python=3.12
    ```

3. **Activar el entorno virtual**:

    - En Windows:

        ```bash
        venv\Scriptsctivate
        ```

    - En macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

### Instalar las dependencias

Con el entorno virtual activado, instala las dependencias necesarias utilizando `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Contenido de `requirements.txt`

Este archivo debe incluir las siguientes dependencias:

```txt
contourpy==1.3.1
cycler==0.12.1
fonttools==4.56.0
joblib==1.4.2
kiwisolver==1.4.8
matplotlib==3.10.1
numpy==2.2.4
packaging==24.2
pandas==2.2.3
pillow==11.1.0
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2
xgboost==3.0.0
```

## Cómo ejecutar el código

Una vez que las dependencias estén instaladas y el entorno virtual esté activado, puedes ejecutar el código directamente desde un script Python o desde un archivo Jupyter Notebook.

1. **Ejecutar desde un archivo Python**:

    El código está guardado en un archivo llamado `main.py`, simplemente ejecuta:

    ```bash
    python main.py
    ```

2. **Ejecutar desde un Jupyter Notebook**:

    Si prefieres trabajar en un entorno interactivo, puedes convertir el código a un archivo de Jupyter Notebook (`.ipynb`) y ejecutarlo paso a paso.

## Resultados esperados

El código entrenará los modelos con los datos simulados y los evaluará utilizando métricas de **MAE** y **RMSE**. Los resultados de cada modelo se imprimirán en la consola. También se generarán gráficos de dispersión para visualizar la comparación entre las ventas reales y las ventas predichas para los tres enfoques.

---

## Conclusión

Este proyecto es un ejemplo de cómo usar técnicas de regresión para predecir ventas utilizando modelos como **Random Forest**, **XGBoost** y **Stacking**. A través de la optimización de hiperparámetros, la evaluación de los modelos y la visualización de los resultados, se demuestra cómo construir un flujo de trabajo completo para predecir una variable continua.

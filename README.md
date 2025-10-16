# improved-genetic-algorithm

El artículo ["An improved genetic algorithm for solving the helicopter routing problem with time window in post-disaster rescue"](https://www.aimspress.com/article/doi/10.3934/mbe.2023699) discute el problema de ruteo de vehículos con ventanas de tiempo (VRPTW) con el fin de diseñar rutas para helicópteros en operaciones de rescate posteriores a un desastre. Para ello, se propuso la implementación de un algoritmo genético mejorado (IGA), en donde se implementa una búsqueda local y global para la mejora de la explotación y la explotación respectivamente, además de una estrategia de inicialización cooperativa para una alta calidad y diversidad en la población inicial.

A este algoritmo se le añaden las siguientes modificaciones, con el fin de mejorar su convergencia:

* 
* 

## Instalación y ejecución ⚙️

1. Instala [Python](https://www.python.org/downloads).
2. Descarga el repositorio localmente.
3. Instala las dependencias necesarias.
    ```sh
    pip install -r requirements.txt
    ```
4. Para ejecutar las benchmarks de Solomon en el código *replicado* de IGA, desde el directorio raíz del proyecto, corre el siguiente comando:
    ```sh
    python -m experiments.run_tests
    ```
5. Para ejecutar las benchmarks de Solomon en el código *modificado* de IGA, desde el directorio raíz del proyecto, corre el siguiente comando:
    ```sh
    python -m experiments.run_tests_IGAm
    ```

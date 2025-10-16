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

## Archivos principales 🗎

* [*Instancias Solomon*](/data/solomon_dataset/)

    Instancias Solomon para problemas VRPTW, obtenidos de [Kaggle](https://www.kaggle.com/datasets/masud7866/solomon-vrptw-benchmark). Estas contienen 6 tipos de benchmarks:

    * **C1:** Clientes agrupados, horizonte de cronograma corto
    * **C2:** Clientes agrupados, horizonte de cronograma largo
    * **R1:** Clientes aleatorios, horizonte de cronograma corto
    * **R2:** Clientes aleatorios, horizonte de cronograma largo
    * **RC1:** Combinación de clientes aleatorios y clientes agrupados, horizonte de cronograma corto
    * **RC2:** Combinación de clientes aleatorios y clientes agrupados, horizonte de cronograma largo

    Las mejores soluciones encontradas se encuentran en el siguiente [link](https://www.sintef.no/projectweb/top/vrptw/100-customers).

* [*Improved Genetic Algorithm (IGA)*](/src/IGA.py)

    Este archivo contiene la implementación del algoritmo genético mejorado, en base al pseudocódigo del artículo mencionado en la descripción. En dicha implementación, se asumieron y simplificaron detalles no especificados en el artículo, particularmente lo siguiente:

    * El tiempo de viaje entre sitios es nulo. Este tiempo puede ser considerado como proporcional a la distancia recorrida al modificar la constante ```travelTime``` en el código.
    * Las ventanas de tiempo son blandas. El tiempo de servicio no necesariamente se completa dentro de la ventana de tiempo.
    * No se puede comenzar el tiempo de servicio si la ventana de tiempo no ha comenzado.
    * No se realiza una distinción entre helicópteros de rescate y helicópteros de transporte.

* [*Improved Genetic Algorithm Modificado (IGAm)*](/src/IGAm.py)

    Este archivo contiene la implementación de las modificaciones a IGA.

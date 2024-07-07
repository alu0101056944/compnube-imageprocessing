#!/bin/sh
##
## Nombre del trabajo que se va a crear
##
#PBS -N gpu_prop_v10
##
## Cola donde se lanzará el trabajo
##
#PBS -q batch
##
## Se ha de solicitar la ejecución en uno de los nodos con GPU's instalada
##
#PBS -l nodes=verode10
##
## Tiempo máximo de ejecución del trabajo. El formato es HH:MM:SS.
##
#PBS -l walltime=01:00:00
##
## Se van a volcar a fichero tanto la salida estándar como la salida de errores
##
#PBS -k oe

# Se debe desactivar el cacheado de los kernel compilados
# pues se hace en el directorio ~/.nv/ComputeCache/ que usado
# por NFS no funciona debido a los bloqueos.
export CUDA_CACHE_DISABLE=1

# Se ha de indicar la ruta absoluta al programa a ejecutar
~/CUDA/gpu_prop/gpu_prop

# Estrategia de Despliegue

## Contenerización

Todos los servicios se empaquetan como imágenes **Docker** siguiendo un
esquema de build multi-etapa, minimizando el tamaño final de la imagen y
evitando incluir herramientas de compilación en la imagen de producción.

## Orquestación

El despliegue se gestiona mediante **Helm charts** sobre Kubernetes, lo que
permite parametrizar cada entorno (desarrollo, staging, producción) sin
duplicar manifiestos YAML. Cada servicio define sus propios recursos
(CPU, memoria) y políticas de auto-scaling en su chart correspondiente.

## Estrategia de Release

Utilizamos despliegues **blue-green** para los servicios críticos de cara
al usuario: se levanta la nueva versión en paralelo a la vigente, se
verifica su salud mediante health checks automáticos, y solo entonces se
redirige el tráfico. Esto permite un rollback inmediato ante cualquier
anomalía, simplemente revirtiendo el enrutamiento.

## Entornos

Existen tres entornos formales: **desarrollo** (despliegue automático en
cada merge a main), **staging** (réplica fiel de producción usada para
pruebas de aceptación) y **producción**. Ningún cambio llega a producción
sin haber pasado antes por staging con aprobación manual de un
responsable técnico.

## Monitoreo y Observabilidad

Todos los servicios exponen métricas en formato Prometheus, visualizadas
en dashboards de **Grafana**. Las alertas críticas (latencia elevada, tasa
de errores, saturación de recursos) se enrutan automáticamente a un canal
de Slack del equipo de guardia mediante Alertmanager.

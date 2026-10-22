# Estrategia de Testing

## Testing Unitario

Todo el código de negocio debe contar con pruebas unitarias escritas con
**pytest**. Se exige una cobertura mínima del **80%** en cada módulo antes
de poder mergear un pull request a la rama principal, verificada
automáticamente por el pipeline de CI.

## Testing de Integración

Para probar la interacción real con bases de datos, colas de mensajería y
otros servicios externos, utilizamos **testcontainers**, que levanta
instancias efímeras de PostgreSQL, Redis y Kafka en contenedores Docker
durante la ejecución de los tests, evitando así depender de entornos
compartidos inestables.

## Testing de Contrato

Los microservicios que se comunican entre sí mediante APIs REST deben
mantener un contrato de API versionado (OpenAPI/Swagger). Se ejecutan
pruebas de contrato automatizadas para detectar cambios que rompan la
compatibilidad con los consumidores antes de llegar a producción.

## Integración Continua

El pipeline de CI corre en **GitHub Actions** y se ejecuta en cada push a
una rama de feature. Las etapas incluyen: linting (ruff), type-checking
(mypy), tests unitarios, tests de integración y un análisis de
vulnerabilidades de dependencias (dependabot + safety).

## Testing en Producción

Además de las pruebas previas al despliegue, aplicamos técnicas de
"testing en producción" controladas: despliegues canary donde una nueva
versión recibe inicialmente el 5% del tráfico real antes de escalar al
100%, con rollback automático si las métricas de error superan el umbral
definido.

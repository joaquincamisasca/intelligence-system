# Arquitectura del Sistema

## Visión General

Nuestra plataforma está construida bajo una arquitectura de microservicios,
diseñada para escalar de forma independiente cada componente según su carga
real. Cada microservicio es responsable de un dominio de negocio acotado y
se comunica con los demás mediante APIs REST o eventos asíncronos.

## Stack Tecnológico

Para los endpoints REST utilizamos **FastAPI** como framework principal, ya
que su soporte nativo para async/await y su integración con Pydantic para
validación de datos lo hacen ideal para servicios de alto rendimiento.

La persistencia de datos transaccionales se maneja con **PostgreSQL**, elegido
por su robustez, soporte de transacciones ACID y el amplio ecosistema de
herramientas de migración disponibles (Alembic).

Para el cacheo de resultados frecuentes y el manejo de sesiones utilizamos
**Redis**, que también actúa como broker de mensajes para tareas en segundo
plano en combinación con Celery.

La comunicación asíncrona entre microservicios (por ejemplo, notificaciones
de cambios de estado en un pedido) se resuelve mediante **Kafka**, que nos
permite desacoplar productores y consumidores de eventos y mantener un
historial reproducible de los mensajes.

## Despliegue

Todos los servicios corren containerizados sobre **Kubernetes**, lo que nos
permite escalar horizontalmente cada microservicio de forma independiente y
aplicar políticas de auto-scaling basadas en métricas de CPU y memoria.

## Principios de Diseño

1. **Independencia de despliegue**: cada microservicio se despliega sin
   necesidad de coordinar releases con otros equipos.
2. **Tolerancia a fallos**: los servicios deben degradar de forma controlada
   (circuit breakers) en vez de propagar fallos en cascada.
3. **Observabilidad desde el diseño**: todo servicio nuevo debe exponer
   métricas y logs estructurados desde el día uno.

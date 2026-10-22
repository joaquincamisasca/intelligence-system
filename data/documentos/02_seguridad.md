# Prácticas de Seguridad

## Autenticación y Autorización

Todos los servicios expuestos externamente utilizan **OAuth2** como esquema
de autenticación, con tokens **JWT** de corta duración (15 minutos) y
refresh tokens rotativos almacenados de forma segura en el cliente.

La autorización a nivel de recurso se implementa mediante un modelo de
control de acceso basado en roles (RBAC), donde cada endpoint declara
explícitamente los roles habilitados para invocarlo.

## Gestión de Secretos

Ninguna credencial, API key o certificado se almacena en el código fuente
ni en variables de entorno planas en producción. Todos los secretos se
gestionan a través de **HashiCorp Vault**, que provee rotación automática
de credenciales de base de datos y auditoría de accesos.

## Comunicación Segura

Toda comunicación entre servicios, tanto interna como externa, viaja
cifrada mediante **TLS 1.3**. Los certificados internos se renuevan
automáticamente mediante cert-manager dentro del clúster de Kubernetes.

## Rate Limiting y Protección ante Abuso

Los endpoints públicos aplican límites de tasa (rate limiting) configurados
por IP y por usuario autenticado, con umbrales diferenciados según el tipo
de operación (lectura vs. escritura). Las peticiones que excedan el límite
reciben un código de estado 429.

## Auditoría

Cada operación de escritura sobre datos sensibles (usuarios, pagos,
permisos) genera un registro de auditoría inmutable, almacenado por
separado de la base de datos operacional, con el usuario responsable,
timestamp y el diff de los cambios realizados.

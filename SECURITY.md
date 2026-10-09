# Security Policy

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Email **enoch.das@gmail.com** with what you found, how to reproduce it, and the commit you tested. You'll get an acknowledgement within a few days.

## Deployment notes

The `docker/docker-compose.yml` stack is a **local development setup**, not a production configuration:

| Component | Development default | Before exposing beyond your machine |
|-----------|--------------------|-------------------------------------|
| Elasticsearch | `xpack.security.enabled=false` | Enable security and TLS, create scoped users |
| Grafana | admin password `admin` | Set `GF_SECURITY_ADMIN_PASSWORD` from a secret |
| Redis, Kafka | no authentication | Enable ACLs / SASL and keep them on a private network |
| Spring Boot | `/actuator/prometheus` and health details exposed | Restrict actuator endpoints to the monitoring network |

The dataset and all derived data stay out of git (`data/` is ignored); secrets belong in environment variables, never in `application.properties`.

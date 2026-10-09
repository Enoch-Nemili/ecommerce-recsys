# Changelog

All notable changes to this project. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-09

First release: the full pipeline from raw clickstream to a load-tested serving API, validated on a 5% user sample (49,222 users, ~5M events) of the Taobao UserBehavior dataset.

### Added
- **Offline pipeline (Spark)**: sessionization, item/user features, co-visitation candidates, and leakage-free training examples from a time-based history/future split.
- **Ranker**: LightGBM, AUC 0.9542, Recall@10 0.9480, NDCG@10 0.9299; exported to ONNX for Java serving.
- **Two-tower embeddings** (PyTorch, GPU) with best-checkpoint tracking (val AUC 0.9583) and a FAISS ANN index.
- **Feature and candidate stores in Redis**: 972,562 items and 49,221 users, loaded in about 6 seconds with pipelining.
- **Java 17 / Spring Boot serving**: `/recommend` (Redis candidates + ONNX ranker + co-visitation tiebreaker) and `/search` (Elasticsearch over 972,562 documents).
- **Streaming**: Kafka producer/consumer updating Redis features in real time.
- **Load test and observability**: Locust (p95 10-12 ms on `/recommend`, ~90 req/s, 0 failures) and an auto-provisioned Prometheus + Grafana dashboard.
- CI: Python lint and a Java compile check on every push.

### Fixed (found while building)
- Feature-table fan-out from grouping on item and category.
- Label leakage from computing features over the full time range.
- An unbounded cross-join that would have produced ~55 billion rows at real scale.
- Corrupted timestamps skewing the time split; the cutoff now uses quantiles and drops implausible rows.

### Changed
- The ONNX model path is now configurable via `RANKER_MODEL_PATH` instead of a hard-coded local path.

[1.0.0]: https://github.com/Enoch-Nemili/ecommerce-recsys/releases/tag/v1.0.0

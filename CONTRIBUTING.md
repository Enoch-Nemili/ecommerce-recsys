# Contributing

Issues and pull requests are welcome.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd docker && docker compose up -d redis      # plus kafka, elasticsearch, prometheus, grafana as needed
```

See the README's Setup section for the dataset and the pipeline stages.

## Before opening a pull request

```bash
pip install ruff
ruff check src/pipeline
mvn -B -q -f src/serving/recsys-service/pom.xml compile
```

If you change features, labels or the ranker, re-run stages 01-03 and paste the before/after AUC, Recall@10 and NDCG@10 into the PR. Keep the time-based split: never compute features from the "future" window.

## Conventions

- One branch and one pull request per change; reference issues with `Closes #N`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, ...).
- Every reported metric comes from a real run; label anything synthetic as synthetic.
- Security issues: see [SECURITY.md](SECURITY.md).

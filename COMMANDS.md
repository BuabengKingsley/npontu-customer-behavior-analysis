# Command-Line Commands Used

Every command-line step used to build, run, and reproduce this project, in the order they'd be run from scratch. Run from the repository root unless noted.

## 1. Environment setup

```bash
pip install -r requirements.txt
```

Installs everything needed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, xgboost, shap, openpyxl, jupyter/notebook, and the Elasticsearch client.

If installing packages individually instead:

```bash
pip install openpyxl
pip install xgboost shap
pip install "elasticsearch>=8.15,<9"
```

The Elasticsearch client version matters: a plain `pip install elasticsearch` pulls the 9.x client by default, which is incompatible with the 8.15.0 server used below (`BadRequestError` on the API version header). It must be pinned to `<9`.

## 2. Generate the synthetic dataset

```bash
python generate_data.py
```

Produces `customers.csv`, `sessions.csv`, `product_interactions.csv`, `purchases.csv`, and `ecommerce_dataset.xlsx`. Deterministic (seed 42) — re-running reproduces the exact same data.

## 3. Run the main analysis notebook

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb --ExecutePreprocessor.timeout=600
```

Executes every cell top to bottom and saves the outputs back into the notebook file. Takes a few minutes — the 227k-row groupbys, 5-fold XGBoost cross-validation, and SHAP explainability are the slow steps, not a hang. Produces `customer_features_scored.csv` as a side effect (used by the Elasticsearch notebook).

To open it interactively instead of running headless:

```bash
jupyter notebook analysis.ipynb
```

## 4. Run the Elasticsearch (big data tool) component

Requires Docker Desktop running first.

```bash
# Start a local single-node Elasticsearch instance
docker run -d --name npontu-es -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0

# Confirm it's responding (wait a few seconds after the run command first)
curl -s http://localhost:9200
```

If the container already exists from a previous run (don't recreate it, just restart it):

```bash
docker ps -a --filter name=npontu-es
docker start npontu-es
```

Then execute the notebook (must be run after `analysis.ipynb`, since it consumes `customer_features_scored.csv`):

```bash
jupyter nbconvert --to notebook --execute --inplace elasticsearch_demo.ipynb --ExecutePreprocessor.timeout=300
```

To stop/remove the container when done:

```bash
docker stop npontu-es
docker rm npontu-es
```

## 5. Version control

```bash
git add <files>
git commit -m "<message>"
git push origin main
```

Standard commits throughout — see the repository's commit history for the actual sequence (data generation → analysis notebook → Elasticsearch component → documentation).

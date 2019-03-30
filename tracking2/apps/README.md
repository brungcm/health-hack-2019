# Realtime People Counter - Apps

Apps built to consume object tracker/detector.
The apps is responsable for business logic.


## Tracker

```bash
python -m apps.tracker_counter --config-file ./config/tracker_no_bq_config.json
```
python -m apps.tracker_counter --config-file ./config/secrets/secret_tracker_no_bq_config.json


## Evaluator

```bash
python -m apps.evaluator --config-file ./config/tracker_eval.json
```


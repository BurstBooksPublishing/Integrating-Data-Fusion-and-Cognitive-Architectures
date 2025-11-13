import json, time
from sklearn.linear_model import SGDClassifier  # incremental learner
from sklearn.metrics import log_loss
import joblib
# ---- Batch retrain (run on scheduled cluster job) ----
def batch_retrain(feature_path, model_registry_path):
    X, y = load_parquet_features(feature_path)     # feature store snapshot
    # full validation and counterfactual checks (not shown)
    model = SGDClassifier(max_iter=1000)           # fresh model
    model.fit(X, y)                                # full-batch training
    val_loss = validate_model(model, X_val, y_val) # gated acceptance
    if val_loss < threshold:
        joblib.dump({'model': model, 'meta': {'loss': val_loss, 'time': time.time()}},
                    model_registry_path)          # sign and register
    return val_loss

# ---- Streaming update (consumer loop) ----
from confluent_kafka import Consumer
def streaming_update(kafka_conf, topic, model_registry_path):
    # load current model artifact; fallback to cold-start
    artifact = joblib.load(model_registry_path)
    model = artifact['model']
    c = Consumer(kafka_conf)
    c.subscribe([topic])
    window = []
    while True:
        msg = c.poll(timeout=1.0)
        if msg is None: continue
        if msg.error():
            continue
        event = json.loads(msg.value())
        if not schema_validate(event):   # defensive gate
            continue
        x, y = featurize_event(event)
        window.append((x, y))
        if len(window) >= 32:            # minibatch update
            Xb = [t[0] for t in window]; yb = [t[1] for t in window]
            model.partial_fit(Xb, yb, classes=classes)  # incremental update
            telemetry.emit({'stream_update': True, 'loss': log_loss(yb, model.predict_proba(Xb))})
            window.clear()
        # periodic checkpoint of streaming state to archive (summary)
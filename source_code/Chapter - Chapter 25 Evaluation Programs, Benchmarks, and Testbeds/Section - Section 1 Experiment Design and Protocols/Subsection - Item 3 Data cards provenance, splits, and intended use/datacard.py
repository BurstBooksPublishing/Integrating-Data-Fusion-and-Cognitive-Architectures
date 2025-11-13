import hashlib, json, os, time
import pandas as pd
from sklearn.model_selection import train_test_split

def file_sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# Load manifest describing raw files and metadata
manifest = pd.read_csv('manifest.csv')  # contains columns: filepath, sensor_id, ts, label

# Compute provenance hashes (expensive: do once, cache results)
manifest['sha256'] = manifest['filepath'].apply(file_sha256)

# Deterministic stratified split using seed and class label if present
seed = 42
alpha, beta = 0.7, 0.15
train_val, test = train_test_split(manifest, test_size=(1-alpha-beta),
                                   stratify=manifest['label'], random_state=seed)
train, val = train_test_split(train_val, test_size=beta/(alpha+beta),
                              stratify=train_val['label'], random_state=seed)

# Assemble data card
data_card = {
  "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "provenance": {
    "manifest_source": "manifest.csv",
    "manifest_hash": file_sha256('manifest.csv'),
    "files": train[['filepath','sha256']].to_dict(orient='records')[:5]  # sample
  },
  "lineage": [
    {"step":"ingest","code_commit":"abc123","container":"sha256:..."},
    {"step":"preprocess","params":{"res":"640x480","normalize":True}}
  ],
  "splits": {
    "seed": seed,
    "train_indices": train.index.tolist(),
    "val_indices": val.index.tolist(),
    "test_indices": test.index.tolist()
  },
  "intended_use": {
    "tasks":["object-detection","tracking"],
    "forbidden":["face_recognition","surveillance_abuse"],
    "license":"CC-BY-4.0",
    "notes":"Contains synthetic augmentations; see lineage for flags."
  }
}

with open('data_card.json','w') as f:
    json.dump(data_card, f, indent=2)
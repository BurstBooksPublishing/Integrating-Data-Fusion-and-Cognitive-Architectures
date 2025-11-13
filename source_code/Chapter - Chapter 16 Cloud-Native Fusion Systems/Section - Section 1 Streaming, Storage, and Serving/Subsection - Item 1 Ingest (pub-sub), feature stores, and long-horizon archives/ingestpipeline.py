#!/usr/bin/env python3
# pip install kafka-python psycopg2-binary boto3 redis
import json, os, time
from kafka import KafkaConsumer  # simple consumer
import psycopg2                     # offline/online feature upsert
import boto3                        # archive to S3
import redis                        # online kv store

KAFKA_TOPIC='sensor-events'
S3_BUCKET='fusion-archive'
PG_DSN=os.environ['PG_DSN']         # postgres connection string
REDIS_URL=os.environ.get('REDIS_URL','redis://localhost:6379')

# initialize clients
consumer = KafkaConsumer(KAFKA_TOPIC, bootstrap_servers='kafka:9092',
                         value_deserializer=lambda b: json.loads(b.decode()))
s3 = boto3.client('s3')
pg = psycopg2.connect(PG_DSN)
redis_client = redis.from_url(REDIS_URL)

UPSERT_SQL = """
INSERT INTO features (key, ts, feat_json, provenance)
VALUES (%s,%s,%s,%s)
ON CONFLICT (key) DO UPDATE
SET ts = EXCLUDED.ts, feat_json = EXCLUDED.feat_json, provenance = EXCLUDED.provenance;
"""

for msg in consumer:
    ev = msg.value
    key = ev['track_id']           # partitioning key
    ts = ev['timestamp']
    # compute simple features (example)
    feat = {'x': ev['x'], 'y': ev['y'], 'speed': ev.get('speed',0.0)}
    prov = {'offset': msg.offset, 'topic': msg.topic, 'producer': ev.get('source_id')}

    # upsert to postgres (offline canonical store)
    with pg.cursor() as cur:
        cur.execute(UPSERT_SQL, (key, ts, json.dumps(feat), json.dumps(prov)))
    pg.commit()

    # update low-latency cache for cognition (online store)
    redis_client.hmset(f"feat:{key}", feat)  # short TTL may be applied in ops

    # archive raw event to S3 (partition by date)
    date_prefix = time.strftime('%Y/%m/%d', time.gmtime(ts/1000))
    s3.put_object(Bucket=S3_BUCKET,
                  Key=f"{date_prefix}/{key}/{ts}.json",
                  Body=json.dumps({'event':ev, 'prov':prov}))
    # minimal inline comments; production should add retries, idempotency, and metrics
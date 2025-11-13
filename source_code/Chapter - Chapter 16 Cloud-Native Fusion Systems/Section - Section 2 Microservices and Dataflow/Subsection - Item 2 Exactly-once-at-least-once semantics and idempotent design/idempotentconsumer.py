import json, time
from confluent_kafka import Consumer
import psycopg2, psycopg2.extras

# Kafka consumer config
c = Consumer({'bootstrap.servers':'kafka:9092','group.id':'fusion','enable.auto.commit':False})
c.subscribe(['tracks'])

# Postgres connection (pooling recommended in production)
conn = psycopg2.connect("dbname=fusion user=fuser host=db sslmode=disable")
conn.autocommit = False

while True:
    msg = c.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        continue  # log and handle broker errors
    data = json.loads(msg.value())            # { "msg_id": "...", "track": { ... } }
    msg_id = data['msg_id']
    track = data['track']
    try:
        with conn.cursor() as cur:
            # dedupe table has PRIMARY KEY(msg_id)
            cur.execute("INSERT INTO processed_msgs (msg_id, received_at) VALUES (%s, now()) ON CONFLICT DO NOTHING", (msg_id,))
            if cur.rowcount == 0:
                # already processed; skip domain work
                conn.commit()
                c.commit(message=msg)  # safe to ack; duplicate ignored
                continue
            # perform idempotent domain update (upsert by track_id)
            cur.execute("""
              INSERT INTO tracks (track_id, state, updated_at) VALUES (%s, %s, now())
              ON CONFLICT (track_id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()
            """, (track['id'], json.dumps(track)))
            conn.commit()
        c.commit(message=msg)  # ack after durable commit
    except Exception as e:
        conn.rollback()
        # log, sleep/backoff, and allow retry; message not acknowledged
        time.sleep(0.5)
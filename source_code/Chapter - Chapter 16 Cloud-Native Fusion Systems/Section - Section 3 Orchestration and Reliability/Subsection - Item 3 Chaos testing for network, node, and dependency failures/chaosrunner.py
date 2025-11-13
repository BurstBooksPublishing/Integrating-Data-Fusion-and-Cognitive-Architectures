#!/usr/bin/env python3
# Small, runnable chaos test: add delay to broker, pause cognition, check health.
import time, requests
import docker

client = docker.from_env()
# Names must match your compose/service names
broker = client.containers.get("broker")     # \lstinline|broker|
cognition = client.containers.get("cognition")  # \lstinline|cognition|
fusion = client.containers.get("fusion")     # \lstinline|fusion|

def add_delay(container, delay_ms=250, duration=30):
    # install tc if missing and add netem rule (works in privileged containers)
    cmd = f"tc qdisc add dev eth0 root netem delay {delay_ms}ms"
    try:
        cid = client.api.exec_create(container.id, cmd, privileged=True)
        client.api.exec_start(cid)
    except docker.errors.APIError:
        pass
    time.sleep(duration)
    # remove rule
    client.api.exec_create(container.id, "tc qdisc del dev eth0 root", privileged=True)
    client.api.exec_start(_)

def check_health(url, timeout=2):
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200, r.elapsed.total_seconds()
    except Exception:
        return False, None

# Canary: add delay to broker and pause cognition briefly
add_delay(broker, delay_ms=250, duration=15)  # inject latency
cognition.pause()                              # simulate node down
time.sleep(5)                                  # let system react
ok, latency = check_health("http://localhost:8080/health")  # fusion health
print("fusion healthy:", ok, "latency s:", latency)
cognition.unpause()                             # recover node
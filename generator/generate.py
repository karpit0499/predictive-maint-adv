import os
import json
import time
import random
from datetime import datetime, timezone

from google.cloud import pubsub_v1, bigquery

PROJECT = os.environ["PROJECT"]
publisher = pubsub_v1.PublisherClient()
TOPIC = publisher.topic_path(PROJECT, "sensor-telemetry")
bq = bigquery.Client(project=PROJECT)
FAILURES = f"{PROJECT}.maintenance.failure_events"

N_MACHINES = int(os.environ.get("N_MACHINES", "60"))
SLEEP = float(os.environ.get("SLEEP", "1.0"))        # seconds between rounds — LEAVE AT 1.0
DECAY = float(os.environ.get("DECAY", "0.0001"))     # ~43-minute machine lifecycle
VIB_OFFSET = float(os.environ.get("VIB_OFFSET", "0.0"))   # Phase 10 injects drift with this

machines = {f"M{i:03d}": {"health": random.uniform(0.7, 1.0)} for i in range(N_MACHINES)}


def reading(mid, m):
    wear = 1.0 - m["health"]                          # 0 = healthy, 1 = dead
    return {
        "machine_id": mid,
        "event_time": datetime.now(timezone.utc).isoformat(),
        "temperature": round(60 + 40 * wear + random.gauss(0, 2), 2),            # heats up as it wears
        "vibration":   round(0.2 + 2.5 * (wear ** 2) + VIB_OFFSET
                             + random.gauss(0, 0.05), 3),                        # spikes late
        "rpm":         round(1500 - 200 * wear + random.gauss(0, 15), 1),
        "pressure":    round(30 - 8 * wear + random.gauss(0, 0.5), 2),
        "voltage":     round(230 + random.gauss(0, 1.5), 2),                     # pure noise (a distractor)
    }


def log_failure(mid):
    """GROUND TRUTH. In a real plant this row comes from the maintenance system,
    not from the sensors. Keeping it in its own table is what lets Phase 4 label
    honestly instead of thresholding a column it also trains on."""
    errs = bq.insert_rows_json(FAILURES, [{
        "machine_id": mid,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }])
    if errs:
        print("failure_events insert errors:", errs)


print(f"publishing {N_MACHINES} machines → {TOPIC}")
print(f"DECAY={DECAY}  SLEEP={SLEEP}  VIB_OFFSET={VIB_OFFSET}")
failures = 0
while True:
    for mid, m in machines.items():
        publisher.publish(TOPIC, json.dumps(reading(mid, m)).encode("utf-8"))
        m["health"] -= random.uniform(0.5, 1.5) * DECAY * (1 + (1 - m["health"]))  # decays faster when worn
        if random.random() < 0.002:                    # occasional sudden fault
            m["health"] -= random.uniform(0.05, 0.15)
        if m["health"] <= 0.05:                        # FAILURE → log it, then replace the machine
            failures += 1
            print(f"{mid} FAILED (#{failures}) — logged to failure_events, machine replaced")
            log_failure(mid)
            m["health"] = random.uniform(0.85, 1.0)
    time.sleep(SLEEP)
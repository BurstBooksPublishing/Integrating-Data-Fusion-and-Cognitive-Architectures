import time, json
# simple safety check - replace with certified check in production
def safety_check(plan): return plan.get("violates_safety")==False

def dispatch_to_substation(plan): 
    # send to PLC/SCADA via secure channel (placeholder)
    print("Dispatching:", json.dumps(plan))

def log_audit(entry): 
    # append to tamper-evident log store (placeholder)
    print("AUDIT:", json.dumps(entry))

def arbitration_and_dispatch(plan, U_auto, U_human, c_f, c_o, tau=0.7):
    # compute authority weights (simple rule)
    if c_f < tau:
        w_a = 0.0
        w_h = 1.0
    else:
        w_h = min(1.0, c_o)
        w_a = 1.0 - w_h
    U_total = w_h*U_human + w_a*U_auto
    # safety veto
    if not safety_check(plan):
        log_audit({"time":time.time(), "action":"vetoed", "reason":"safety"})
        return False
    # latency guard (ensure human available within budget)
    T_max = plan.get("latency_budget", 5.0)
    start = time.time()
    # simulated decision interval
    time.sleep(0.05)
    if time.time() - start > T_max:
        # execute safe default
        default = plan.get("safe_default")
        dispatch_to_substation(default)
        log_audit({"time":time.time(), "action":"fallback", "default":default})
        return True
    # commit if utility positive and authorized
    if U_total > 0:
        dispatch_to_substation(plan)
        log_audit({"time":time.time(), "U_total":U_total, "weights":{"w_h":w_h,"w_a":w_a}})
        return True
    log_audit({"time":time.time(), "action":"rejected", "U_total":U_total})
    return False

# Example invocation (replace with real message fields)
plan = {"id":"shed_42","violates_safety":False,"latency_budget":2.0}
arbitration_and_dispatch(plan, U_auto=0.7, U_human=0.6, c_f=0.82, c_o=0.9)
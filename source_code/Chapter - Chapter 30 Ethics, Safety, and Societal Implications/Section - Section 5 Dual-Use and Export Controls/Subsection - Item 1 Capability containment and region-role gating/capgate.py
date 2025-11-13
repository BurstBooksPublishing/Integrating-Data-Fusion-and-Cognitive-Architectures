import hmac, hashlib, time, logging
# Example policy database (would be stored/served securely)
POLICY = {
    "synthesize_trajectory": {
        "roles": {"analyst", "safety_officer"},
        "regions": {"onprem", "secure_edge"},
        "sensitivity": 10
    }
}
# Region clearance levels
REGION_CLEARANCE = {"onprem": 20, "secure_edge": 15, "public_cloud": 5}
SECRET = b"replace_with_secret_key"  # HMAC key in secure vault

def verify_token(token, expected_claims):
    # token format: msg||hex_hmac; simple example for illustration
    try:
        msg, hex_sig = token.rsplit("|", 1)
    except ValueError:
        return False
    sig = hmac.new(SECRET, msg.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, hex_sig) and msg == expected_claims

def gate_request(role, region, capability, token):
    # brief provenance and policy checks
    if capability not in POLICY:
        logging.warning("Unknown capability: %s", capability)
        return False, "unknown_capability"
    p = POLICY[capability]
    if role not in p["roles"]:
        return False, "role_not_allowed"
    if region not in p["regions"]:
        return False, "region_not_allowed"
    if p["sensitivity"] > REGION_CLEARANCE.get(region, 0):
        return False, "insufficient_region_clearance"
    if not verify_token(token, f"{role}|{region}|{capability}"):
        return False, "invalid_token"
    return True, "allowed"

# Example usage: wrapped cognitive endpoint
def synthesize_trajectory(request):
    ok, reason = gate_request(request["role"], request["region"],
                              "synthesize_trajectory", request["token"])
    if not ok:
        logging.info("Request denied: %s", reason)  # audit log
        return {"status": "denied", "reason": reason}
    # proceed with capability action (placeholder)
    return {"status": "ok", "trajectory": [0,1,2]}  # real planner call here
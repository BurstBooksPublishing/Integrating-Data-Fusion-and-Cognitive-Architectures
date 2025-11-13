from flask import Flask, request, jsonify, g
import jwt, time, logging

# Shared secret for example; replace with public-key verification in production.
SECRET = "secret-key"
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def verify_jwt(token):
    # decode token and return claims; raises on invalid
    return jwt.decode(token, SECRET, algorithms=["HS256"])

def resource_labels(resource_id):
    # example mapping; in real systems query metadata store.
    return {"tenant": "tenantA", "region": "eu-west-1"}

@app.before_request
def enforce_tenant_region():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "missing token"}), 401
    token = auth.split(None, 1)[1]
    try:
        claims = verify_jwt(token)
    except Exception:
        return jsonify({"error": "invalid token"}), 401
    # pull request resource and operation
    res = request.view_args.get("resource_id") or request.args.get("resource")
    labels = resource_labels(res)
    # policy checks: tenant match and region residency enforcement
    if claims.get("tenant") != labels["tenant"]:
        # provenance log for audit
        logging.info("DENY tenant-mismatch ts=%s subj=%s res=%s", int(time.time()), claims.get("sub"), res)
        return jsonify({"error": "tenant mismatch"}), 403
    if claims.get("region") != labels["region"]:
        logging.info("DENY region-mismatch ts=%s subj=%s res=%s", int(time.time()), claims.get("sub"), res)
        return jsonify({"error": "region constraint"}), 403
    # attach provenance for downstream services
    g.provenance = {"actor": claims.get("sub"), "tenant": claims.get("tenant"), "region": claims.get("region"), "ts": int(time.time())}

@app.route("/fuse/", methods=["POST"])
def fuse(resource_id):
    # process uses g.provenance; cognitive modules record this in trace store
    return jsonify({"status": "accepted", "resource": resource_id, "provenance": g.provenance})

if __name__ == "__main__":
    app.run(port=8080)
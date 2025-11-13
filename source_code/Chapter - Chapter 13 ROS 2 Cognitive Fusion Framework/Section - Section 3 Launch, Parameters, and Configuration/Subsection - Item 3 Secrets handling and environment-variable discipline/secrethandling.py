#!/usr/bin/env python3
import os, stat, requests
import rclpy
from rclpy.node import Node

SECRET_ENV = "FUSION_API_TOKEN"
SECRET_FILE = "/run/secrets/fusion_api_token"
MIN_LEN = 24  # minimum characters

def read_secret():
    # try env var first (convenience), else file mounted by orchestrator
    token = os.environ.get(SECRET_ENV)
    if token:
        return token
    if os.path.exists(SECRET_FILE):
        st = os.stat(SECRET_FILE)
        # ensure file is not world-readable
        if st.st_mode & (stat.S_IRWXO) != 0:
            raise PermissionError("Secret file has insecure permissions")
        with open(SECRET_FILE, "r") as f:
            return f.read().strip()
    raise FileNotFoundError("No secret found")

def use_and_scrub_token(token):
    try:
        if len(token) < MIN_LEN:
            raise ValueError("Secret too short")
        # use token in a single request header (do not log token)
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.post("https://cognition-service.example/api/score",
                          json={"input": "frame_meta"}, headers=headers, timeout=2.0)
        r.raise_for_status()
        return r.json()
    finally:
        # overwrite and delete token to reduce memory residuals
        token = "0" * len(token)
        del token

class FusionClient(Node):
    def __init__(self):
        super().__init__("fusion_client")
        try:
            tok = read_secret()
            resp = use_and_scrub_token(tok)
            # handle response without exposing token data
            self.get_logger().info("Inference result received")
        except Exception as e:
            self.get_logger().error(f"Secret access or call failed: {e}")
            # fail closed; escalate for operator action

def main():
    rclpy.init()
    node = FusionClient()
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
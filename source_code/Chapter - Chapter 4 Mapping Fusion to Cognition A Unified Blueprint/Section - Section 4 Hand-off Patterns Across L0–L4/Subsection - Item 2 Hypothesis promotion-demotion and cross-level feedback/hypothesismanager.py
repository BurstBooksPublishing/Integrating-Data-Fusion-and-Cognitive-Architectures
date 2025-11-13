#!/usr/bin/env python3
import math, time, logging
logging.basicConfig(level=logging.INFO)

class Hypothesis:
    def __init__(self, id, prior_logodds=0.0):
        self.id = id
        self.logodds = prior_logodds
        self.provenance = []            # list of (sensor,likelihood,timestamp)
        self.state = "candidate"        # candidate/promoted/demoted

    def update(self, sensor, p_z_given_H, p_z_given_notH):
        # accumulate log-odds per Eq. (1)
        delta = math.log(p_z_given_H) - math.log(p_z_given_notH)
        self.logodds += delta
        self.provenance.append((sensor, p_z_given_H, p_z_given_notH, time.time()))
        return delta

class HypothesisManager:
    def __init__(self, tau_plus=3.0, tau_minus=-2.0):
        self.hypotheses = {}
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def handle_measurement(self, hid, sensor, p_h, p_not_h):
        h = self.hypotheses.setdefault(hid, Hypothesis(hid))
        delta = h.update(sensor, p_h, p_not_h)
        logging.info("Hyp %s updated by %+.3f -> logodds %.3f", hid, delta, h.logodds)
        self._apply_policy(h)

    def _apply_policy(self, h):
        if h.logodds > self.tau_plus and h.state != "promoted":
            h.state = "promoted"
            self._on_promote(h)
        elif h.logodds < self.tau_minus and h.state != "demoted":
            h.state = "demoted"
            self._on_demote(h)

    def _on_promote(self, h):
        logging.info("PROMOTE %s; issuing sensor feedback", h.id)
        self._publish_feedback(h.id, action="increase_rate", targets=["radar_beam_3"])
        # audit record would be persisted here

    def _on_demote(self, h):
        logging.info("DEMOTE %s; releasing resources", h.id)
        self._publish_feedback(h.id, action="loosen_gating", targets=["track_assoc"])

    def _publish_feedback(self, hid, action, targets):
        # stub: replace with ROS2/DDS publish or RPC to lower-level services
        logging.info("FEEDBACK %s -> %s on %s", hid, action, targets)

# Simple run
if __name__ == "__main__":
    mgr = HypothesisManager()
    mgr.handle_measurement("H1", "radar", 0.9, 0.1)   # strong support
    mgr.handle_measurement("H1", "ais", 0.2, 0.8)     # contradicting evidence
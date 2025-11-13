import json, time, math
import numpy as np

class AlertDecider:
    def __init__(self, thresholds, abstain_uc=0.3):
        # thresholds: dict mapping cohort -> score threshold (0..1)
        self.thresholds = thresholds
        self.abstain_uc = abstain_uc
    def calibrate_score(self, raw_score, calib_fn):
        # simple interface; calib_fn maps raw->calibrated
        return calib_fn(raw_score)
    def decide(self, patient_id, cohort, raw_score, epistemic_uc, metadata, calib_fn=lambda x:x):
        s = self.calibrate_score(raw_score, calib_fn)
        # abstain to human if epistemic uncertainty high
        if epistemic_uc >= self.abstain_uc:
            return {"action":"escalate_human","reason":"high_uncertainty","score":s}
        th = self.thresholds.get(cohort, self.thresholds.get("default", 0.5))
        if s >= th:
            # emit alert with rationale snippet
            alert = {"action":"alert","score":s,"threshold":th,
                     "rationale":{"features":metadata.get("features","na")}}
            self._log_event(patient_id, alert, metadata)
            return alert
        self._log_event(patient_id, {"action":"no_alert","score":s}, metadata)
        return {"action":"no_alert","score":s}
    def _log_event(self, pid, event, metadata):
        rec = {"ts":time.time(),"patient":pid,"event":event,"meta":metadata}
        print(json.dumps(rec))  # replace with structured telemetry sink

# Example usage
dec = AlertDecider({"default":0.7,"icu":0.65})
print(dec.decide("p123","default", raw_score=0.78, epistemic_uc=0.1,
                 metadata={"features":["hr_trend","lactate"]}))
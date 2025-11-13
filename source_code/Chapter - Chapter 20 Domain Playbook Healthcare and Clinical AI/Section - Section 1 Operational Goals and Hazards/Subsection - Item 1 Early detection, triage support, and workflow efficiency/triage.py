import math, json, queue, time
# simple calibrated scorers (placeholders); real models replace these
def vitals_score(vitals): return (vitals['hr']>120)*1.5 + (vitals['rr']>22)*1.0
def labs_score(labs): return (labs.get('lactate',0)>2.0)*2.0
def notes_score(notes): return ('infection' in notes.lower())*1.2

def log_lr(lr): return math.log(max(lr,1e-6))  # log-likelihood ratio safe log

def compute_posterior(prior, vitals, labs, notes, weights):
    # compute log-odds then convert to probability
    logodds = math.log(prior/(1-prior))
    logodds += weights['vitals']*log_lr(vitals_score(vitals))
    logodds += weights['labs']*log_lr(labs_score(labs))
    logodds += weights['notes']*log_lr(notes_score(notes))
    odds = math.exp(logodds)
    return odds/(1+odds)

# simulated queue and run
alert_q = queue.Queue()
weights = {'vitals':0.8,'labs':1.0,'notes':0.6}
prior = 0.02  # baseline prevalence estimate

def process_patient_stream(pid, vitals, labs, notes):
    p = compute_posterior(prior, vitals, labs, notes, weights)
    rationale = {'pid':pid,'p_post':p,'components':{'vitals':vitals,'labs':labs,'notes':notes}}
    if p>0.25:  # triage threshold tuned to clinician capacity
        alert = {'time':time.time(),'pid':pid,'priority':'high','p':p,'rationale':rationale}
        alert_q.put(alert)
    # record for audit
    print(json.dumps(rationale))

# example call
process_patient_stream('patient-123', {'hr':130,'rr':24}, {'lactate':2.8}, "Possible infection, fever noted")
# consumer would dequeue and integrate with EHR workflow
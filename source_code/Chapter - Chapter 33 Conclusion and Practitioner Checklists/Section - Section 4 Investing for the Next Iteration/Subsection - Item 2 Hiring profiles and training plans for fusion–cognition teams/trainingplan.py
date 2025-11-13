import pandas as pd
# load competencies: rows are people, cols are skills, values in [0,1]
skills = pd.read_csv("skill_matrix.csv", index_col=0)  # person,skill values
# required thresholds per skill (user-specified)
required = {"state_estimation":0.8, "data_association":0.7, "cognitive_modeling":0.6,
            "ros2_deploy":0.7, "assurance_testing":0.75}
weights = {k:1.0 for k in required}  # importance weights

# compute team coverage per Eq. (1)
coverage = skills[list(required)].mean(axis=0)
gaps = pd.Series({k:max(0, required[k]-coverage[k]) for k in required})

# prioritize trainings by weighted gap magnitude
priorities = (gaps * pd.Series(weights)).sort_values(ascending=False)
print("Coverage:\n", coverage.round(2))
print("\nPrioritized training requirements:\n", priorities)

# simple assignment: find people with lowest competency in top skill and recommend training
top_skill = priorities.index[0]
candidates = skills[top_skill].sort_values().head(3)  # three weakest
recommendations = [{"person":p, "skill":top_skill, "current":skills.loc[p, top_skill]} 
                   for p in candidates.index]
print("\nTraining recommendations (top skill):\n", recommendations)
# further steps: schedule 2-week focused module or pair-program with senior mentor
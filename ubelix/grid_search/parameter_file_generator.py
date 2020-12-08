import numpy as np
import json
import os
import re


estimators = [50, 70, 100, 200, 500]
depths = [10, 30, 60, 150, 300]

parameters = json.load(open("no_features.json", "r"))
parameters['first_order_feature'] = False
parameters['GLCM_features'] = False
parameters['experiment_name'] = "tree_search"
json.dump(parameters, open("no_features.json", 'w+'))

jobscript = open('jobscript.sh', 'r')
jobscript = jobscript.readlines()


for estimator in estimators:
    for depth in depths:
        new_job = open(parameters['experiment_name'] + '.sh', 'w+')
        for line in jobscript:
            line = re.sub('tree_50_10', parameters['experiment_name'], line.rstrip())
            line = re.sub('feature_calculation', parameters['experiment_name'], line.rstrip())
            new_job.write(line + '\n')
            new_job.close()
        parameters['n_estimators'] = estimator
        parameters['max_depth'] = depth
        parameters['experiment_name'] = 'tree_' + str(estimator) + '_' + str(depth)
        json.dump(parameters, open(parameters['experiment_name'] + '.json', 'w+'))
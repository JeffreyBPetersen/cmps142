import math
import random

def generate_instance(dimensionality):
  instance = []
  for i in range(dimensionality):
    instance.append(random.choice([-1,1]))
  return instance

def generate_instances(quantity, dimensionality):
  instances = []
  for i in range(quantity):
    instances.append(generate_instance(dimensionality))
  return instances

def get_instance_label(instance, mode):
  if mode == "a":
    return 0 if instance[0] == -1 else 1
  elif mode == "b":
    total = 0
    for i in range(len(instance)):
      total += instance[i]
    return 1 if total > 0 else 0
  elif mode == "c":
    total = 0
    for i in range(len(instance)):
      total += instance[i]
    total += random.choice([-4,-3,-2,-1,0,1,2,3,4])
    return 1 if total > 0 else 0
  else:
    print "ERROR: mode not found in get_instance_label"

def initialize_weights(dimensionality):
  return [0 for i in range(dimensionality + 1)]
    
def calculate_hypothesis(weights, instance):
  hypothesis = weights[0]
  for i in range(len(instance)):
    hypothesis += weights[i+1] * instance[i]
  return hypothesis

def get_hypothesis_label(hypothesis):
  return 0 if hypothesis < 0.5 else 1
  
def update_weights_on_instance(learning_rate, weights, instance, label):
  hypothesis = calculate_hypothesis(weights, instance)
  new_weights = [weights[0] + learning_rate * (label - hypothesis)]
  for i in range(len(instance)):
    new_weights.append(weights[i+1] + learning_rate * (label - hypothesis) * instance[i])
  return new_weights

def run_experiment_a(training, testing, learning_rate):
  training_labels = [get_instance_label(instance, "a") for instance in training]
  testing_labels = [get_instance_label(instance, "a") for instance in testing]
  weights = initialize_weights(len(training[0]))
  epochs = 0
  prediction_errors = 0
  total_prediction_errors = 0
  correct_on_test_set = False
  while not correct_on_test_set:
    epochs += 1
    prediction_errors = 0
    for instance, label in zip(training, training_labels):
      if get_hypothesis_label(calculate_hypothesis(weights, instance)) != label:
        prediction_errors += 1
        weights = update_weights_on_instance(learning_rate, weights, instance, label)
    correct_on_test_set = True
    for instance, label in zip(testing, testing_labels):
      if get_hypothesis_label(calculate_hypothesis(weights, instance)) != label:
        correct_on_test_set = False
        break
    total_prediction_errors += prediction_errors
    print "epoch #", epochs, ", prediction errors: ", prediction_errors
  print "total prediction errors: ", total_prediction_errors
  
def run_experiment_b(training, testing, learning_rate):
  training_labels = [get_instance_label(instance, "b") for instance in training]
  testing_labels = [get_instance_label(instance, "b") for instance in testing]
  weights = initialize_weights(len(training[0]))
  epochs = 0
  prediction_errors = 0
  total_prediction_errors = 0
  accuracy = 0
  while (accuracy != 1 and epochs < 100) or (accuracy < 0.95 and epochs < 1000):
    epochs += 1
    prediction_errors = 0
    for instance, label in zip(training, training_labels):
      if get_hypothesis_label(calculate_hypothesis(weights, instance)) != label:
        prediction_errors += 1
        weights = update_weights_on_instance(learning_rate, weights, instance, label)
    hits = 0
    for instance, label in zip(testing, testing_labels):
      if get_hypothesis_label(calculate_hypothesis(weights, instance)) == label:
        hits += 1
    accuracy = hits / len(testing)
    total_prediction_errors += prediction_errors
    print "epoch #", epochs, ", prediction errors: ", prediction_errors
  print "total prediction errors: ", total_prediction_errors
  if accuracy < 1:
    "imperfect accuracy after >100 epochs, accuracy: ", accuracy
  if epochs == 1000:
    print "could not correctly predict testing labels after 1000 epochs"
  
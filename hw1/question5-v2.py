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
  dot_product = weights[0]
  for i in range(len(instance)):
    dot_product += weights[i+1] * instance[i]
  return 1 / (1 + math.e**(-dot_product))

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
    
def log_likelihood(weights, instances, labels):
  likelihood = 0
  for instance, label in zip(instances, labels):
    hypothesis = calculate_hypothesis(weights, instance)
    likelihood += label * math.log(hypothesis) + (1 - label) * math.log(1 - hypothesis)
  return likelihood
    
def run_experiment_c(training, testing, learning_rate):
  training_labels = [get_instance_label(instance, "c") for instance in training]
  testing_labels = [get_instance_label(instance, "c") for instance in testing]
  weights = initialize_weights(len(training[0]))
  weights_history = []
  epochs = 0
  while epochs < 2:
    epochs += 1
    for instance, label in zip(training, training_labels):
      if get_hypothesis_label(calculate_hypothesis(weights, instance)) != label:
        weights = update_weights_on_instance(learning_rate, weights, instance, label)
      weights_history.append(weights)
  average_weights = initialize_weights(len(training[0]))
  average_2nd_epoch_weights = initialize_weights(len(training[0]))
  final_weights = weights_history[-1]
  for i in range(len(weights_history)):
    for j in range(len(weights_history[i])):
      average_weights[j] += weights_history[i][j]
      if i >= len(training):
        average_2nd_epoch_weights[j] += weights_history[i][j]
  for i in range(len(average_weights)):
    average_weights[i] /= len(training)
    average_2nd_epoch_weights[i] /= len(training)
  avg_weights_likelihood = log_likelihood(average_weights, testing, testing_labels)
  avg_weights_2_likelihood = log_likelihood(average_2nd_epoch_weights, testing, testing_labels)
  final_weights_likelihood = log_likelihood(final_weights, testing, testing_labels)
  print "average weights: ", average_weights, ", log likelihood: ", avg_weights_likelihood
  print "average 2nd epoch weights: ", average_2nd_epoch_weights, ", log likelihood: ", avg_weights_2_likelihood
  print "final weights: ", final_weights, ", log likelihood: ", final_weights_likelihood

def main():
  training = generate_instances(500,11)
  testing = generate_instances(500,11)
  learning_rate = float(raw_input("Set learning rate within (0.0, 1.0]: "))
  experiment_selection = raw_input("Select experiment - enter a, b, or c: ")
  if experiment_selection == "a":
    run_experiment_a(training, testing, learning_rate)
  elif experiment_selection == "b":
    run_experiment_b(training, testing, learning_rate)
  elif experiment_selection == "c":
    run_experiment_c(training, testing, learning_rate)
  else:
    print "ERROR: experiment selection not found"

main()
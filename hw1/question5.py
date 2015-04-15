import random
import math

# named constants
training_size = 500
testing_size = 500
dimensionality = 11
learning_rate = 1 # set within range [0,1]

## implement update_weights(weights)

# takes two integers
def generate_instances(quantity, dimensionality):
  instances = []
  for instance in range(quantity):
    instances.append([])
    for component in range(dimensionality):
      instances[instance].append(random.choice([-1,1]))
  return instances

# takes unlabelled instances and one of ["a", "b", "c"]
def label_instances(instances, labelling_mode):
  labelled_instances = []
  for instance in instances:
    if labelling_mode == "a":
      labelled_instances.append((instance, 1 if instance[0] == 1 else 0))
    elif labelling_mode == "b":
      labelled_instances.append((instance, 1 if sum(instance) > 0 else 0))
    elif labelling_mode == "c":
      labelled_instances.append((instance, 1 if sum(instance) + random.randrange(-4,4) >= 0 else 0))
    else:
      print "invalid labelling mode"
  return labelled_instances

# takes weights and unlabelled instance
def calculate_hypothesis(weights, instance):
  dot_product = weights[0]
  for i in range(dimensionality):
    dot_product += weights[i+1] * instance[i]
  return 1 / (1 + math.exp(-dot_product))
  
# takes weights and unlabelled instance
def calculate_prediction(hypothesis):
  return 0 if hypothesis < 0.5 else 1

# takes learning rate, weights, labelled instance, and predicted value for label
def update_weights(learning_rate, weights, instance, prediction):
  new_weights = []
  new_weights.append(weights[0] + learning_rate * (instance[1] - prediction))
  for i in range(dimensionality):
    new_weights.append(weights[i+1] + learning_rate * (instance[1] - prediction) * instance[0][i])
  return new_weights

print "UCSC Machine Learning hw1q5\n"

# generate instances as [components]
training = generate_instances(training_size, dimensionality)
testing = generate_instances(testing_size, dimensionality)
# set learning_rate
learning_rate = float(raw_input("Set learning rate within (0.0, 1.0]: "))
# label training instances as ([components], label)
experiment_selection = raw_input("Select experiment - enter a, b, or c: ")
training = label_instances(training, experiment_selection)
testing = label_instances(testing, experiment_selection)
# initialize weights as [(dimensionality + 1) 0s]
weights = []
for i in range(dimensionality+1):
  weights.append(0)

if experiment_selection == "a":
  print "- experiment a -"
  epochs = 0
  accuracy = 0
  while accuracy < 1 and epochs < 1000:
    epochs += 1
    hits = 0.0
    misses = 0.0
    for instance in training:
      prediction = calculate_prediction(calculate_hypothesis(weights, instance[0]))
      weights = update_weights(learning_rate, weights, instance, prediction)
    for instance in testing:
      prediction = calculate_prediction(calculate_hypothesis(weights, instance[0]))
      if prediction == instance[1]:
        hits += 1
      else:
        misses += 1
    accuracy = hits / (hits + misses)
    print "epoch #", epochs, ": accuracy = ", accuracy
  print "epochs = ", epochs
elif experiment_selection == "b":
  print "- experiment b -"
  epochs = 0
  accuracy = 0
  while accuracy < 1 and epochs < 1000:
    epochs += 1
    hits = 0.0
    misses = 0.0
    for instance in training:
      prediction = calculate_prediction(calculate_hypothesis(weights, instance[0]))
      weights = update_weights(learning_rate, weights, instance, prediction)
    for instance in testing:
      prediction = calculate_prediction(calculate_hypothesis(weights, instance[0]))
      if prediction == instance[1]:
        hits += 1
      else:
        misses += 1
    accuracy = hits / (hits + misses)
    print "epoch #", epochs, ": accuracy = ", accuracy
  print "epochs = ", epochs
elif experiment_selection == "c":
  print "- experiment c -"
  epochs = 0
  accuracy = 0
  weights_history = []
  average_weights = []
  average_weights_2nd_epoch = []
  for i in range(dimensionality):
    average_weights.append(0)
    average_weights_2nd_epoch.append(0)
  while epochs < 2:
    epochs += 1
    hits = 0.0
    misses = 0.0
    for instance in training:
      prediction = calculate_prediction(calculate_hypothesis(weights, instance[0]))
      weights = update_weights(learning_rate, weights, instance, prediction)
      weights_history.append(weights)
  for i in range(1000):
    for j in range(dimensionality):
      average_weights[j] += weights_history[i][j]
      if i > 499:
        average_weights_2nd_epoch[j] += weights_history[i][j]
  for i in range(dimensionality):
    average_weights[i] /= 1000
    average_weights_2nd_epoch[i] /= 500
  print "average weights: ", average_weights
  print "average 2nd epoch weights: ", average_weights_2nd_epoch
else:
  print "ERROR: invalid experiment selection"

raw_input("--- press any key to end program ---")

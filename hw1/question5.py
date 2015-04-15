import random

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
def calculate_prediction(weights, instance):
  prediction = weights[0]
  for i in range(dimensionality):
    prediction += weights[i+1] * instance[i]
  return 0 if prediction < 0.5 else 1

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
# label training instances as ([components], label)
experiment_selection = raw_input("Select experiment - enter a, b, or c: ")
training = label_instances(training, experiment_selection)
testing = label_instances(testing, experiment_selection)
# initialize weights as [(dimensionality + 1) 0s]
weights = []
for i in range(dimensionality+1):
  weights.append(0)

for instance in training:
  print instance
  prediction = calculate_prediction(weights, instance[0])
  print prediction
  weights = update_weights(learning_rate, weights, instance, prediction)
  print weights

raw_input("--- press any key to end program ---")

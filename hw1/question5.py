import random

# named constants
training_size = 500
testing_size = 500
dimensionality = 11
learning_rate = 1 # set within range [0,1]

## implement label_instance(mode) # mode is 'a', 'b', or 'c'
## implement update_weights(weights)

def generate_instances(quantity, dimensionality):
  instances = []
  for instance in range(quantity):
    instances.append([])
    for component in range(dimensionality):
      instances[instance].append(random.choice([-1,1]))
  return instances

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

print "UCSC Machine Learning hw1q5\n"

# generate instances as [components]
training = generate_instances(training_size, dimensionality)
testing = generate_instances(testing_size, dimensionality)
# label training instances as ([components], label)
labelling_mode = raw_input("Select experiment - enter a, b, or c: ")
training = label_instances(training, labelling_mode)
testing = label_instances(testing, labelling_mode)

for instance in training:
  print instance

raw_input("--- press any key to end program ---")

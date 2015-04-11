import random

print "UCSC Machine Learning hw1q5\n"

training = []
for i in range(500):
  training.append([])
  for j in range(11):
    training[i].append(random.choice([-1,1]))

print training

raw_input("--- press any key to end program ---")

import math

class Neuron:
  def __init__(self):
    self.inputs = []
    self.outputs = []
    self.weights = [0]
    self.is_input_node = None
    
  def get_output(self, input_vector):
    total = self.weights[0]
    for index, input in enumerate(input_vector):
      total += self.weights[index+1] * input
    return 1 / (1 + math.exp(-total))

class Neuralnet:
  # network_map is a list of lists representing node connections
  #   each list is of the incoming connections to the node with matching index
  # number_of_inputs is the number of nodes to initialize as input nodes
  def __init__(self, network_map, number_of_inputs):
    # initialize nodes with connections
    self.nodes = []
    self.number_of_inputs = number_of_inputs
    for node_index, connections in enumerate(network_map):
      self.nodes.append(Neuron())
      self.nodes[node_index].is_input_node = node_index < number_of_inputs
      for connection in connections:
        self.nodes[node_index].inputs.append(connection)
        self.nodes[node_index].weights.append(0)
        self.nodes[connection].outputs.append(node_index)
    self.learning_rate = 0.1
  
  # inputs are mapped to the nodes with matching indices
  def prop(self, inputs):
    outputs = []
    for input in inputs:
      outputs.append(input)
    for node in self.nodes:
      if not node.is_input_node:
        inputs_to_node = []
        for input in node.inputs:
          inputs_to_node.append(outputs[input])
        outputs.append(node.get_output(inputs_to_node))
    return outputs
    
  def backprop(self, outputs, target):
    output_errors = {} # partial E to a
    new_weights = [[weight for weight in node.weights] for node in self.nodes] # partial E to w_
    x = outputs[-1]
    output_errors[len(self.nodes)-1] = (
      -math.exp(x)*((target - 1) * math.exp(x) + target) / math.pow(math.exp(x) + 1, 3))
    indices = range(self.number_of_inputs, len(self.nodes) - 1)
    indices.reverse()
    for index in indices:
      total = 0
      for output in self.nodes[index].outputs:
        weight_index = self.nodes[output].inputs.index(index) + 1
        total += output_errors[output] * self.nodes[output].weights[weight_index]
      total *= outputs[index] * (1 - outputs[index])
      output_errors[index] = total
    for index, node in enumerate(self.nodes):
      if not index < self.number_of_inputs:
        for index_2, weight in enumerate(node.weights):
          total = output_errors[index]
          if index_2 != 0:
            total *= outputs[node.inputs[index_2 - 1]]
          new_weights[index][index_2] = new_weights[index][index_2] - self.learning_rate * total
    for index, node in enumerate(self.nodes):
      for index_2, weight in enumerate(node.weights):
        node.weights[index_2] = new_weights[index][index_2]
  
  def fit(self, X_train, y_train):
    for index, x in enumerate(X_train):
      outputs = prop(x)
      if outputs[-1] != y_train[index]:
        backprop(outputs, y_train[index])
    
  def predict(self, X_test):
    return [self.prop(x)[-1] for x in X_test]

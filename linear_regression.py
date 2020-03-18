class NeuralNetwork:
  def __init__(self):
    #mx + b - 2 weights: m and b
    self.m = 4
    self.b = 1
    
  def test(self, inp): #y_hat
    return self.m * inp + self.b;
  
  def train(self, epochs, inp, ex_out, learning_rate):
    for epoch in range(epochs):
      #print("epoch " + str(epoch) + ": " + str(self.backpropogate(inp, ex_out, learning_rate)))
      #uncomment top line to print measure after each epoch
      self.backpropogate(inp, ex_out, learning_rate)
  
  def backpropogate(self, inp, ex_out, learning_rate):
    error = 0
    for i in range(len(inp)):
      curr_in = inp[i];
      curr_ex = ex_out[i];
      y_hat = self.test(curr_in);
      dEdyhat = 2 * (y_hat - curr_ex)
      dm = dEdyhat * curr_in
      db = dEdyhat
      newM = self.m - dm * learning_rate
      newB = self.b - db * learning_rate
      self.m = newM	#Repeating cycle with newM and newB equal to m and b respectively
      self.b = newB
      error = (curr_ex - y_hat) ** 2
    return error

def main():
  network = NeuralNetwork()
  x = [1, 2, 3, 4, 5, 6]
  y = [3, 4, 7, 9, 12, 13]
  #print(network.test(x[0]))
  network.train(10000, x, y, learning_rate = 0.02)  #change this to change training details: epochs, 
  print("y = " + str(network.m) + "x + " + str(network.b))
  
main()
  

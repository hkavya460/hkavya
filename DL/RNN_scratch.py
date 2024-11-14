import numpy as np
import string

# Utility function for one-hot encoding
#one hot encoding for convert the cateogorical value to numerical values 
def string_one_hot_encode(inputs=np.ndarray) -> np.ndarray:
    char_index = {char: i for i, char in enumerate(string.ascii_uppercase)}     #charcter to integer conversion 
    one_hot_inputs = []

    for rows in inputs:
        one_hot_list = []
        for char in rows:
            if char.upper() in char_index:
                one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
                one_hot_vector[char_index[char.upper()]] = 1
                one_hot_list.append(one_hot_vector)
        # Concatenate the one-hot vectors for this row into a single array
        one_hot_inputs.append(np.concatenate(one_hot_list, axis=1))

    return np.array(one_hot_inputs)

# Input layer:random initalization of wieghts and bias for inputs
class InputLayer:
    def __init__(self, inputs: np.ndarray, hidden_size: int) -> None:
        self.inputs = inputs
        self.u = np.random.uniform(low=0, high=1, size=(hidden_size, inputs.shape[1]))
        self.delta_u = np.zeros_like(self.u)

    def get_inputs(self, time_step: int) -> np.ndarray:
        return self.inputs[time_step].reshape(-1, 1)

    def weighted_sum(self, time_step: int) -> np.ndarray:
        inputs = self.get_inputs(time_step)
        return self.u @ inputs

    def calculate_deltas_per_step(self, time_step: int, delta_weighted_sum: np.ndarray) -> None:
        self.delta_u += delta_weighted_sum @ self.get_inputs(time_step).T

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.u -= learning_rate * self.delta_u

# Hidden layer
#intalization of wieghts and biases for the hidden layer 
class HiddenLayer:
    def __init__(self, vocab_size: int, size: int, max_time_steps: int) -> None:
        self.w = np.random.uniform(low=0, high=1, size=(size, size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(max_time_steps, size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.next_delta_activation = np.zeros(shape=(size, 1))
        self.delta_w = np.zeros_like(self.w)

    def get_hidden_state(self, time_step: int) -> np.ndarray:
        if time_step < 0 or time_step >= self.states.shape[0]:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    def set_hidden_state(self, time_step: int, hidden_state: np.ndarray) -> None:
        self.states[time_step] = hidden_state

    #activating the revious hidden states with respect to time for passing the output to next input 

    def activate(self, weighted_input: np.ndarray, time_step: int) -> np.ndarray:
        previous_hidden_state = self.get_hidden_state(time_step - 1)
        weighted_hidden_state = self.w @ previous_hidden_state
        weighted_sum = weighted_input + weighted_hidden_state + self.bias
        activation = np.tanh(weighted_sum)
        self.set_hidden_state(time_step, activation)
        return activation

    def calculate_deltas_per_step(self, time_step: int, delta_output: np.ndarray) -> np.ndarray:
        delta_activation = delta_output + self.next_delta_activation
        delta_weighted_sum = delta_activation * (1 - self.get_hidden_state(time_step)) ** 2

        self.next_delta_activation = self.w.T @ delta_weighted_sum
        self.delta_w += delta_weighted_sum @ self.get_hidden_state(time_step - 1).T
        self.delta_bias += delta_weighted_sum
        return delta_weighted_sum
#updating the weights after backpropagation 
    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.w -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_bias

# Softmax function to find the probabaility 
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# Output layer using hidden state and weight intalized for the output will be calculated (w_hy) 
class OutputLayer:
    def __init__(self, size: int, hidden_size: int) -> None:
        self.v = np.random.uniform(low=0, high=1, size=(size, hidden_size))
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.states = np.zeros(shape=(size, size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.delta_v = np.zeros_like(self.v)
#using the softmax function which is the higgest among the all will be consider as the output 
    def predict(self, hidden_state: np.ndarray, time_step: int) -> np.ndarray:
        output = self.v @ hidden_state + self.bias
        prediction = softmax(output)
        self.set_state(time_step, prediction)
        return prediction

    def get_state(self, time_step: int) -> np.ndarray:
        return self.states[time_step]

    def set_state(self, time_step: int, prediction: np.ndarray) -> None:
        self.states[time_step] = prediction

    def calculate_deltas_per_step(self, expected: np.ndarray, hidden_state: np.ndarray, time_step: int) -> np.ndarray:
        delta_output = self.get_state(time_step) - expected
        self.delta_v += delta_output @ hidden_state.T
        self.delta_bias += delta_output
        return self.v.T @ delta_output

    def update_weights_and_bias(self, learning_rate: float) -> None:
        self.v -= learning_rate * self.delta_v
        self.bias -= learning_rate * self.delta_bias

# Vanilla RNN
#Functions like feed forward and backpropgation using these function all the parameters wioll be updated 
class VanillaRNN:
    def __init__(self, vocab_size: int, hidden_size: int, alpha: float) -> None:
        self.hidden_layer = HiddenLayer(vocab_size, hidden_size, max_time_steps=26)  # Example max_time_steps
        self.output_layer = OutputLayer(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        self.alpha = alpha

    def feed_forward(self, inputs: np.ndarray) -> OutputLayer:
        self.input_layer = InputLayer(inputs, self.hidden_size)
        for step in range(len(inputs)):
            weighted_input = self.input_layer.weighted_sum(step)
            activation = self.hidden_layer.activate(weighted_input, step)
            self.output_layer.predict(activation, step)
        return self.output_layer

    def backpropagation(self, expected: np.ndarray) -> None:
        num_time_steps = len(expected)
        for step_number in reversed(range(min(num_time_steps, self.hidden_layer.states.shape[0]))):
            hidden_state = self.hidden_layer.get_hidden_state(step_number)
            delta_output = self.output_layer.calculate_deltas_per_step(expected[step_number], hidden_state, step_number)
            delta_weighted_sum = self.hidden_layer.calculate_deltas_per_step(step_number, delta_output)
            self.output_layer.update_weights_and_bias(self.alpha)
            self.hidden_layer.update_weights_and_bias(self.alpha)
            self.input_layer.update_weights_and_bias(self.alpha)

    def loss(self, y_hat: list[np.ndarray], y: list[np.ndarray]) -> float:
        return sum(-np.sum(y[i] * np.log(y_hat[i]) for i in range(len(y))))

    def train(self, inputs: np.ndarray, expected: np.ndarray, epochs: int) -> None:
        for epoch in range(epochs):
            for idx, input in enumerate(inputs):
                y_hats = self.feed_forward(input)
                self.backpropagation(expected[idx].reshape(-1, 1))
                print(f"Loss round :{self.loss([y for y in y_hats.states], expected[idx])}")

if __name__ == "__main__":
    #input 
    inputs = np.array([["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
                       ["Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A"],
                       ["B", "D", "F", "H", "J", "L", "N", "P", "R", "T", "V", "X", "Z", "A", "C", "E", "G", "I", "K", "M", "O", "Q", "S", "U", "W", "Y"],
                       ["M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
                       ["H", "G", "F", "E", "D", "C", "B", "A", "L", "K", "J", "I", "P", "O", "N", "M", "U", "T", "S", "R", "Q", "X", "W", "V", "Z", "Y"]])

    expected = np.array([
        ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A"],
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
        ["C", "E", "G", "I", "K", "M", "O", "Q", "S", "U", "W", "Y", "A", "B", "D", "F", "H", "J", "L", "N", "P", "R", "T", "V", "X", "Z"],
        ["N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
        ["I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "A", "B", "C", "D", "E", "F", "G", "H"]])
#calling the functions 
    one_hot_inputs = string_one_hot_encode(inputs)
    one_hot_expected = string_one_hot_encode(expected)

    rnn = VanillaRNN(vocab_size=len(string.ascii_uppercase), hidden_size=128, alpha=0.0001)
    rnn.train(one_hot_inputs, one_hot_expected, epochs=10)

    new_inputs = np.array([['B', 'C', 'D']])
    for input in string_one_hot_encode(new_inputs):
        predictions = rnn.feed_forward(input)
        output = np.argmax(predictions.states[-1])
        print(output)
        print(string.ascii_uppercase[output])




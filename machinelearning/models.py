import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = self.run(x)
        return 1 if nn.as_scalar(dot_product) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            converged = True  # Assume convergence, will be set to False if a mistake is found
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):
                    # Update weights if misclassified
                    self.w.update(x, nn.as_scalar(y))
                    converged = False  # Training is not converged


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Initialize your model parameters here
        self.learning_rate = .01
        self.weight1 = nn.Parameter(1, 128)
        self.bias1 = nn.Parameter(1, 128)
        self.weight2 = nn.Parameter(128, 64)
        self.bias2 = nn.Parameter(1, 64)
        self.weight3 = nn.Parameter(64, 1)
        self.bias3 = nn.Parameter(1, 1)
        self.parameters = [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3]



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_layer_bias = nn.AddBias(nn.Linear(x, self.weight1), self.bias1)
        first_layer = nn.ReLU(first_layer_bias)
        second_layer_bias = nn.AddBias(nn.Linear(first_layer, self.weight2),self.bias2)
        second_layer = nn.ReLU(second_layer_bias)
        output_layer_linear = nn.Linear(second_layer, self.weight3)
        output_layer = nn.AddBias(output_layer_linear,self.bias3)
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 50
        avg_loss = 1
        while avg_loss > 0.015:  # Adjust the number of epochs as needed
            total_loss = 0
            num_batches = 0
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                grads = nn.gradients(loss, self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(grads[i], -self.learning_rate)
                num_batches += 1
            avg_loss = total_loss / num_batches
            print(avg_loss)
            if avg_loss <= 0.0200:
                break
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 0.5
        self.hidden_size = 200

        # Define parameters and layers
        self.w1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)
        self.params = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first_layer_bias = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        first_layer = nn.ReLU(first_layer_bias)
        linear_bias = nn.Linear(first_layer, self.w2)
        output_layer = nn.AddBias(linear_bias, self.b2)
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SoftmaxLoss(y_hat, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 100
        num_epochs = 5  # Adjust as needed
        print_frequency = 100

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)

                grads = nn.gradients(loss, self.params)

                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)

                num_batches += 1

                if num_batches % print_frequency == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch {epoch + 1}, Batch {num_batches}, Avg Loss: {avg_loss}")

            # Compute validation accuracy after each epoch
            val_accuracy = dataset.get_validation_accuracy()
            print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy}")

        print("Training completed.")
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.initial_weights = nn.Parameter(self.num_chars, 256)
        self.initial_bias = nn.Parameter(1, 256)
        self.x_weights = nn.Parameter(self.num_chars, 256)
        self.h_weights = nn.Parameter(256, 256)
        self.bias = nn.Parameter(1, 256)
        self.output_weights = nn.Parameter(256, len(self.languages))
        self.output_bias = nn.Parameter(1, len(self.languages))
        self.params = [self.initial_weights, self.initial_bias, self.x_weights, self.h_weights,
                       self.bias, self.output_weights, self.output_bias]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        bias_start = nn.AddBias(nn.Linear(xs[0], self.initial_weights), self.initial_bias)
        hidden_state = nn.ReLU(bias_start)
        for char in xs[1:]:
            linear_terms = nn.Add(nn.Linear(char, self.x_weights), nn.Linear(hidden_state, self.h_weights))
            bias = nn.AddBias(linear_terms, self.bias)
            hidden_state = nn.ReLU(bias)
        linear = nn.Linear(hidden_state, self.output_weights)
        output = nn.AddBias(linear, self.output_bias)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(xs)
        return nn.SoftmaxLoss(y_hat, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        size = 100
        accur = 0
        while accur < 0.85:
            for x, y in dataset.iterate_once(size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.learning_rate)
            accur = dataset.get_validation_accuracy()
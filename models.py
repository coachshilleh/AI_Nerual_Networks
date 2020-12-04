import nn
import numpy as np


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

        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        counter = 1
        while counter != 0:
            counter = 0
            for x, y in dataset.iterate_once(1):
                # for each x i want to classify
                prediction = self.get_prediction(x)
                if y.data != prediction:
                    counter += 1
                    self.w.update(x, prediction*-1)











class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(1, 50)
        self.W2 = nn.Parameter(50, 50)
        self.W3 = nn.Parameter(50, 1)
        self.B1 = nn.Parameter(1, 50)
        self.B2 = nn.Parameter(1, 50)
        self.B3 = nn.Parameter(1, 1)
        self.multiplier = 0.005


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """

        # Just used a two layer nerual network as shown on the project website :)

        xm1 =  nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.B1)), self.W2), self.B2)), self.W3), self.B3)
        return xm1

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for epoch in range(0,130):
            print(epoch)
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x,y)
                grad_wrt_W1,  grad_wrt_B1, grad_wrt_W2, grad_wrt_B2,grad_wrt_W3, grad_wrt_B3 = nn.gradients(loss, [self.W1,self.B1,self.W2,self.B2, self.W3,self.B3])
                if nn.as_scalar(loss) > 0.02:
                    self.W1.update(grad_wrt_W1, -1* self.multiplier)
                    self.W2.update(grad_wrt_W2, -1* self.multiplier)
                    self.W3.update(grad_wrt_W3, -1 * self.multiplier)
                    self.B1.update(grad_wrt_B1, -1* self.multiplier)
                    self.B2.update(grad_wrt_B2, -1* self.multiplier)
                    self.B3.update(grad_wrt_B3, -1 * self.multiplier)




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

    ## Also passes 9/10, if it doesnt the first time please try again :)


    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784, 400)
        self.W2 = nn.Parameter(400, 10)
        self.B1 = nn.Parameter(1, 400)
        self.multiplier = 0.3

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
        return nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.B1)), self.W2)

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
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(150):
                loss = self.get_loss(x,y)
                grad_wrt_W1,  grad_wrt_B1, grad_wrt_W2 = nn.gradients(loss, [self.W1,self.B1,self.W2])
                a = dataset.get_validation_accuracy()
                print(a)
                if a < 0.975:
                    self.W1.update(grad_wrt_W1, -1* self.multiplier)
                    self.W2.update(grad_wrt_W2, -1* self.multiplier)
                    self.B1.update(grad_wrt_B1, -1* self.multiplier)

                else:
                    return


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
        self.batchSize = 500
        self.W1 = nn.Parameter(self.num_chars, 50)
        self.W2 = nn.Parameter(50, 50)
        self.W3 = nn.Parameter(50,5)
        self.B1 = nn.Parameter(1, 50)
        self.B2 = nn.Parameter(1, 50)
        self.B3 = nn.Parameter(1, 5)
        self.multiplier = 0.1
        #Hidden
        self.H1 = nn.Parameter(5, 100)
        self.H2 = nn.Parameter(100, 100)
        self.H3 =  nn.Parameter(100, 5)
        self.HB1 = nn.Parameter(1, 100)
        self.HB2 = nn.Parameter(1, 100)
        self.HB3 = nn.Parameter(1,5)
        self.multiplier = 0.1

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

        ## Passes just takes some time :), please wait

        counter = 0
        for i in xs:
            if counter == 0:
                h = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(i, self.W1), self.B1)), self.W2), self.B2)), self.W3), self.B3)
                counter += 1
            else:
                h = nn.Add(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(i, self.W1), self.B1)), self.W2), self.B2)), self.W3), self.B3),
                           nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(h, self.H1), self.HB1)), self.H2), self.HB2)), self.H3), self.HB3))
        return h


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
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:

            for x, y in dataset.iterate_once(self.batchSize):

                loss = self.get_loss(x,y)
                grad_wrt_W1,  grad_wrt_B1, grad_wrt_W2, grad_wrt_B2, grad_wrt_W3, grad_wrt_B3, grad_wrt_H1,  grad_wrt_HB1, grad_wrt_H2, grad_wrt_HB2, grad_wrt_H3, grad_wrt_HB3 = nn.gradients(loss, [self.W1,self.B1,self.W2,self.B2,self.W3,self.B3, self.H1,self.HB1,self.H2,self.HB2, self.H3,self.HB3])
                a = dataset.get_validation_accuracy()
                print(a)
                if a < 0.83:
                    self.W1.update(grad_wrt_W1, -1* self.multiplier)
                    self.W2.update(grad_wrt_W2, -1* self.multiplier)
                    self.W3.update(grad_wrt_W3, -1 * self.multiplier)
                    self.B1.update(grad_wrt_B1, -1* self.multiplier)
                    self.B2.update(grad_wrt_B2, -1* self.multiplier)
                    self.B3.update(grad_wrt_B3, -1 * self.multiplier)


                    self.H1.update(grad_wrt_H1, -1 * self.multiplier)
                    self.H2.update(grad_wrt_H2, -1 * self.multiplier)
                    self.H3.update(grad_wrt_H3, -1 * self.multiplier)
                    self.HB1.update(grad_wrt_HB1, -1 * self.multiplier)
                    self.HB2.update(grad_wrt_HB2, -1 * self.multiplier)
                    self.HB3.update(grad_wrt_HB3, -1 * self.multiplier)
                else:
                    return

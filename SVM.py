import numpy as np
import matplotlib.pyplot as plt


# this is a class for a boundary hyperplane
class Line:
    # w is a list, b  is a scalar
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = b

    # finds predicted y value given current boundary hyperplane
    def find_y(self, point):
        return np.dot(self.w, point.x) + self.b

    # finds distance between a point and the boundary
    def find_dist(self, point):
        return np.abs(self.find_y(point)) / np.linalg.norm(self.w)

    # finds distance between one margin and the boundary
    def find_margin_dist(self):
        return 1/np.linalg.norm(self.w)


# a single data point, note that y is the correct output value, not the predicted one
class Point:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = y
        self.l = 1


class Svm:
    # creates object svm with a margin made from the line class and processed data from the point class
    def __init__(self, margin, data_raw, data_dim):
        self.margin = margin
        processed_data = []

        # converts data to line/point form, assuming that the last value is the correct classified output
        # the first data_dim number of values are assumed to be the x vector
        for i in data_raw:
            processed_data.append(Point(i[0:data_dim], i[-1]))
        self.processed_data = processed_data
        # self.lamda_term = np.array([0 for i in range(data_dim)])

    # Returns a list of point objects that are within the margin of the SVM
    def find_support_vectors(self):
        support_vectors = []
        for i in self.processed_data:
            if self.margin.find_dist(i) <= self.margin.find_margin_dist():
                support_vectors.append(i)
        return support_vectors

    def support_vector_dropout(self):
        support_vectors_dropout = []
        for i in self.processed_data:
            if self.margin.find_dist(i) <= self.margin.find_margin_dist():
                support_vectors_dropout.append(0)
            else:
                support_vectors_dropout.append(1)
        return support_vectors_dropout

    # works only in two dimensions, plots points, margin and boundaries
    def plot_data(self):
        # plotting points
        data_pos = []
        data_neg = []
        for i in self.processed_data:
            if i.y > 0:
                data_pos.append(i.x)
            else:
                data_neg.append(i.x)
        data_pos = np.array(data_pos)
        data_neg = np.array(data_neg)
        x_pos, y_pos = data_pos.T
        x_neg, y_neg = data_neg.T
        plt.scatter(x_pos, y_pos, c="#ff0000")
        plt.scatter(x_neg, y_neg, c="#0000ff")

        # plot line
        slope = self.margin.w[0] / self.margin.w[1] * -1
        intercept = lambda bound_val : (bound_val-self.margin.b)/self.margin.w[1]
        plt.axline((1, slope + intercept(0)), (2, slope * 2 + intercept(0)))
        plt.axline((1, slope + intercept(1)), (2, slope * 2 + intercept(1)), c="000000")
        plt.axline((1, slope + intercept(-1)), (2, slope * 2 + intercept(-1)), c="000000")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.show()

    # calculates the value of the lagrangian, uses what is probably hinge loss
    def calculate_loss(self):
        f = np.linalg.norm(self.margin.w) ** 2 * 0.5
        # g = lambda w, x_i, b, y_i: y_i * (np.dot(w, x_i) + b)
        g_net = 0
        support_vectors = self.find_support_vectors()
        for i in support_vectors:
            g_net = g_net + (self.margin.find_y(i) * i.y - 1) * i.l
        return f + g_net

    # returns a vector of partials of lagrangian with ref to weights
    def calculate_grad_weights(self):
        updates = self.margin.w
        support_vectors = self.find_support_vectors()
        # print("hi")
        for i in support_vectors:
            # print(np.dot(i.y * i.l, i.x))
            updates = updates - np.dot(i.y * i.l, i.x)
        return np.array(updates)

    # returns a vector of partials of lagrangian with ref to weights
    def calculate_grad_bias(self):
        updates = 0
        support_vectors = self.find_support_vectors()
        for i in support_vectors:
            updates = updates - i.y * i.l
        return np.array(updates)

    # Updates the lambda values via gradient descent
    def update_lambda(self, learning_rate_multipliers):
        support_vectors = self.find_support_vectors()
        for i in support_vectors:
            i.l = i.l + np.dot((i.y * (self.margin.find_y(i))) - 1, learning_rate_multipliers)

        '''
        support_vector_dropout = self.support_vector_dropout()
        index = 0
        for i in support_vector_dropout:
            if i == 1:
                i = support_vectors[index].y * (self.margin.find_y(support_vectors[index])) - 1
                index = i + 1
        return np.array(support_vector_dropout)
        '''
        return None

    # trains the svm for n iterations with the inputted learning rates
    def train(self, learning_rate_weights, learning_rate_bias, learning_rate_multipliers, iterations):
        for i in range(iterations):
            print("Training cycle: " + str(i) + "  Loss: " + str(self.calculate_loss()))
            # print(np.dot(self.calculate_grad_weights(), learning_rate_weights))
            self.margin.w = self.margin.w - np.dot(self.calculate_grad_weights(), learning_rate_weights)
            self.margin.b = self.margin.b - (self.calculate_grad_bias() * learning_rate_bias)
            self.update_lambda(learning_rate_multipliers)
            print(self.margin.find_margin_dist())
            # self.plot_data()
        return self

"""
Load the data and train the models, varying the hyperparameters. Output graphs of the results.
"""

import idxreader
from gradientdescent import GradientDescent as gradient_descent_model
from os import path
from sys import stdout
from matplotlib import pyplot as plt


def load_MNIST_data():
    """
    Load the MNIST dataset.
    """
    dir = path.dirname(__file__)
    training_data_filepath = path.join(dir, "..", "MNIST", "digits-train", "train-images-idx3-ubyte")
    training_labels_filepath = path.join(dir, "..", "MNIST", "digits-train", "train-labels-idx1-ubyte")
    test_data_filepath = path.join(dir, "..", "MNIST", "digits-test", "t10k-images-idx3-ubyte")
    test_labels_filepath = path.join(dir, "..", "MNIST", "digits-test", "t10k-labels-idx1-ubyte")

    training_data = idxreader.read(training_data_filepath)
    training_labels = idxreader.read(training_labels_filepath)
    test_data = idxreader.read(test_data_filepath)
    test_labels = idxreader.read(test_labels_filepath)

    return training_data, training_labels, test_data, test_labels


def test_and_plot_gradient_descent(training_data, training_labels, test_data, test_labels, figure):
    """
    Train and test a gradient descent model using various hyperparameters, and plot the results.

    Arguments:
        training_data -- A numpy array representing the data used to train the model
        training_labels -- A numpy array representing the labels for the training data
        test_data -- A numpy array representing the data used to test the model
        test_labels -- A numpy array representing the labels for the test data
        figure -- A matplotlib.pyplot.Figure on which to render the graphs
    """
    # Vary iterations
    print "\nTest varying the iterations"
    print "==========================="

    iterations = [1, 2, 3, 10, 50, 100]
    accuracies = []
    for i, iteration in enumerate(iterations):
        stdout.write("\rTesting model {0} of {1}".format(i+1, len(iterations)))
        stdout.flush()
        gd_model = gradient_descent_model(n_iter=iteration)
        gd_model.train(training_data, training_labels)
        accuracies.append(gd_model.test(test_data, test_labels))
    print "\n"

    mnist_iter_plot = figure.add_subplot(221)
    mnist_iter_plot.set_title("Vary Iterations")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.plot(iterations, accuracies, "o")
    for point in zip(iterations, accuracies):
        mnist_iter_plot.annotate("{0}, {1}".format(*point), xy=point, xytext=(3, 0), textcoords="offset points")

    # Vary learning rate
    print "\nTest varying the learning rate"
    print "=============================="

    learning_rates = [0.1, 0.3, 0.6, 0.9]
    accuracies = []
    for i, learning_rate in enumerate(learning_rates):
        stdout.write("\rTesting model {0} of {1}".format(i+1, len(learning_rates)))
        stdout.flush()
        gd_model = gradient_descent_model(eta0=learning_rate)
        gd_model.train(training_data, training_labels)
        accuracies.append(gd_model.test(test_data, test_labels))
    print "\n"

    mnist_learning_rate_plot = figure.add_subplot(222)
    mnist_learning_rate_plot.set_title("Vary Learning Rate")
    plt.ylabel("Accuracy")
    plt.xlabel("Learning Rate")
    plt.plot(learning_rates, accuracies, "o")
    for point in zip(learning_rates, accuracies):
        mnist_learning_rate_plot.annotate("{0}, {1}".format(*point), xy=point, xytext=(3, 0), textcoords="offset points")

    # Vary regularizer
    print "\nTest varying the regularizer alpha"
    print "=================================="

    alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    accuracies = []

    for i, alpha in enumerate(alphas):
        stdout.write("\rTesting model {0} of {1}".format(i+1, len(alphas)))
        stdout.flush()
        gd_model = gradient_descent_model(penalty="l2", alpha=alpha)
        gd_model.train(training_data, training_labels)
        accuracies.append(gd_model.test(test_data, test_labels))
    print "\n"

    mnist_alpha_plot = figure.add_subplot(223)
    mnist_alpha_plot.set_title("Vary Regularizer Alpha")
    plt.ylabel("Accuracy")
    plt.xlabel("Regularizer Alpha")
    plt.plot(alphas, accuracies, "o")
    for point in zip(alphas, accuracies):
        mnist_alpha_plot.annotate("{0}, {1}".format(*point), xy=point, xytext=(3, 0), textcoords="offset points")

    # Vary loss function
    print "\nTest varying the loss function"
    print "=============================="

    loss_functions = ['log', 'hinge', 'perceptron']
    x_values = []
    accuracies = []

    for i, loss_function in enumerate(loss_functions):
        stdout.write("\rTesting model {0} of {1}".format(i+1, len(loss_functions)))
        stdout.flush()
        gd_model = gradient_descent_model(penalty="l2", loss=loss_function)
        gd_model.train(training_data, training_labels)
        accuracies.append(gd_model.test(test_data, test_labels))
        x_values.append(i)
    print "\n"

    mnist_loss_plot = figure.add_subplot(224)
    mnist_loss_plot.set_title("Vary Loss Function")
    plt.ylabel("Accuracy")
    plt.xlabel("Loss Function")
    plt.plot(x_values, accuracies, "o")
    plt.xticks(x_values, loss_functions)
    for point in zip(x_values, accuracies):
        mnist_loss_plot.annotate("{0}".format(point[1]), xy=point, xytext=(3, 0), textcoords="offset points")


def main():
    mnist_training_data, mnist_training_labels, mnist_testing_data, mnist_testing_labels = load_MNIST_data()

    # Convert the image data into two-dimensional arrays
    # by flattening each image to a 1-dimensional array
    mnist_training_data = mnist_training_data.reshape(len(mnist_training_data), -1)
    mnist_testing_data = mnist_testing_data.reshape(len(mnist_testing_data), -1)

    mnist_gradient_descent_fig = plt.figure(1, figsize=(10,10))
    mnist_gradient_descent_fig.suptitle("MNIST Data: Gradient Descent Model")

    test_and_plot_gradient_descent(mnist_training_data, mnist_training_labels, mnist_testing_data, mnist_testing_labels, mnist_gradient_descent_fig)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

if __name__ == "__main__":
    main()

from neural_network import NeuralNetwork
import numpy as np
from mnist import MNIST


def convert_labels(mnist_labels):
    long_labels = list()
    for i, label in enumerate(training_labels):
        long_labels.append([])
        for j in range(10):
            if j == label:
                long_labels[i].append(1)
            else:
                long_labels[i].append(0)

    return long_labels


def normalize_inputs(mnist_inputs):
    normalized_inputs = list()
    for inputs in mnist_inputs:
        normalized_inputs.append((np.array(inputs) / 255).tolist())

    return normalized_inputs


def predictions_to_proportions(predictions):
    sum_pred = sum(predictions)
    return ["%.2f" % (p/sum_pred) for p in predictions]


num_training_samples = 100000

mnist_data = MNIST("MNIST_dataset_uncompressed")

print("loading training data")
training_images, training_labels = mnist_data.load_training()
training_images = training_images[:num_training_samples]
print("normalizing inputs")
training_inputs = normalize_inputs(training_images[:num_training_samples])

print("parsing labels")
training_outputs = convert_labels(training_labels[:num_training_samples])

# nn = NeuralNetwork.from_existing_model("model")

nn = NeuralNetwork(input_size=784, output_size=10, hidden_size=32, num_hidden_layers=2)

print("Beginning training")
nn.train(training_images, training_outputs, batch_size=40, learning_rate=0.1, iterations=500)

print("Saving model")
nn.save_model("model")

print("loading test data")
test_images, test_labels = mnist_data.load_testing()

print("normalizing inputs")
test_inputs = normalize_inputs(test_images)

print("parsing labels")
test_outputs = convert_labels(test_labels)

print("\n" * 2)
print("testing model on training data:")
np.set_printoptions(precision=5, suppress=True)
for inputs, image in zip(training_inputs[:10], training_images[:10]):
    print(mnist_data.display(image))
    print(predictions_to_proportions(nn.predict(inputs)))

print("\n" * 2)
print("testing model on new data:")
np.set_printoptions(precision=5, suppress=True)
for inputs, image in zip(test_inputs[:10], test_images[:10]):
    print(mnist_data.display(image))
    print(predictions_to_proportions(nn.predict(inputs)))

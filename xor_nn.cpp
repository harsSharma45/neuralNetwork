#include <bits/stdc++.h>
using namespace std;

#define rep(i, a, b) for (int i = a; i < b; ++i)
#define vii vector<double>
#define viii vector<vii>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

double init_weights() {
    return ((double)rand()) / RAND_MAX; // Random weights between 0 and 1
}

double sigmoid(double val) {
    return 1 / (1 + exp(-val));
}

double dSigmoid(double val) {
    return val * (1 - val);
}

void shuffle(vector<int>& vec) {
    random_device rd;
    mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);
}

void print_weights_and_biases(const viii& weights, const vii& biases, const string& name) {
    cout << name << " Weights:\n";
    for (const auto& row : weights) {
        for (double w : row) cout << w << " ";
        cout << "\n";
    }
    cout << name << " Biases:\n";
    for (double b : biases) cout << b << " ";
    cout << "\n";
}


int main() {
    const double lr = 0.1;

    vii hiddenLayer(numHiddenNodes), outputLayer(numOutputs);
    vii hiddenLayerBias(numHiddenNodes), outputLayerBias(numOutputs);
    viii hiddenWeights(numInputs, vii(numHiddenNodes)), outputWeights(numHiddenNodes, vii(numOutputs));

    viii training_inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    viii training_outputs = {{0.0}, {1.0}, {1.0}, {0.0}};

    // Initialize weights and biases
    rep(i, 0, numInputs) rep(j, 0, numHiddenNodes) hiddenWeights[i][j] = init_weights();
    rep(i, 0, numHiddenNodes) rep(j, 0, numOutputs) outputWeights[i][j] = init_weights();
    rep(i, 0, numHiddenNodes) hiddenLayerBias[i] = init_weights();
    rep(i, 0, numOutputs) outputLayerBias[i] = init_weights();

    vector<int> trainingSetOrder = {0, 1, 2, 3};
    int epochs = 1e4;

    rep(epoch, 0, epochs) {
        shuffle(trainingSetOrder);

        for (int idx : trainingSetOrder) {
            // Forward pass: Hidden layer
            rep(j, 0, numHiddenNodes) {
                double activation = hiddenLayerBias[j];
                rep(k, 0, numInputs) activation += training_inputs[idx][k] * hiddenWeights[k][j];
                hiddenLayer[j] = sigmoid(activation);
            }

            // Forward pass: Output layer
            rep(j, 0, numOutputs) {
                double activation = outputLayerBias[j];
                rep(k, 0, numHiddenNodes) activation += hiddenLayer[k] * outputWeights[k][j];
                outputLayer[j] = sigmoid(activation);
            }

            // Backpropagation
            vii deltaOutput(numOutputs), deltaHidden(numHiddenNodes);

            rep(j, 0, numOutputs) {
                double error = training_outputs[idx][j] - outputLayer[j];
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            rep(j, 0, numHiddenNodes) {
                double error = 0.0;
                rep(k, 0, numOutputs) error += deltaOutput[k] * outputWeights[j][k];
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            // Update weights and biases: Output layer
            rep(j, 0, numOutputs) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                rep(k, 0, numHiddenNodes) outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
            }

            // Update weights and biases: Hidden layer
            rep(j, 0, numHiddenNodes) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                rep(k, 0, numInputs) hiddenWeights[k][j] += training_inputs[idx][k] * deltaHidden[j] * lr;
            }
        }

        // Periodically print weights and biases
        if ((epoch + 1) % 1000 == 0) {
            cout << "Epoch " << epoch + 1 << ":\n";
            print_weights_and_biases(hiddenWeights, hiddenLayerBias, "Hidden Layer");
            print_weights_and_biases(outputWeights, outputLayerBias, "Output Layer");
        }
    }

    // Final Outputs
    cout << "Training complete. Final outputs:\n";
    rep(i, 0, numTrainingSets) {
        rep(j, 0, numHiddenNodes) {
            double activation = hiddenLayerBias[j];
            rep(k, 0, numInputs) activation += training_inputs[i][k] * hiddenWeights[k][j];
            hiddenLayer[j] = sigmoid(activation);
        }

        rep(j, 0, numOutputs) {
            double activation = outputLayerBias[j];
            rep(k, 0, numHiddenNodes) activation += hiddenLayer[k] * outputWeights[k][j];
            outputLayer[j] = sigmoid(activation);
        }

        cout << "Input: " << training_inputs[i][0] << ", " << training_inputs[i][1]
             << " -> Predicted Output: " << outputLayer[0]
             << " (Expected: " << training_outputs[i][0] << ")\n";
    }

    return 0;
}



/**
 * @brief Main function to train a simple neural network for XOR problem.
 * 
 * This function initializes the neural network with random weights and biases,
 * trains it using backpropagation, and prints the final outputs after training.
 * 
 * @return int Returns 0 upon successful execution.
 * 
 * The neural network consists of:
 * - An input layer with 2 nodes (for the XOR inputs).
 * - A hidden layer with a specified number of nodes.
 * - An output layer with 1 node (for the XOR output).
 * 
 * The training data consists of the four possible inputs for the XOR function
 * and their corresponding outputs.
 * 
 * The training process involves:
 * - Forward pass: Calculating activations for the hidden and output layers.
 * - Backpropagation: Calculating errors and updating weights and biases.
 * - Periodically printing the weights and biases for monitoring.
 * 
 * After training, the final outputs for the training inputs are printed along
 * with the expected outputs.
 */
#include "headers.h"
#include "loadDataset.h"

using namespace std;

void expOne() {
	vector<vector<float> > labelsTrainV;
	vector<vector<float> > dataTrainV;
	
	cout << "Loading data...\n";
	int trainSize = loadDatasetMNIST("mnist_train.csv", labelsTrainV, dataTrainV);
	vector<float*> labelsTrain;
	for (vector<float> v : labelsTrainV) {
		float* newData = new float[v.size()];
		copy(v.begin(), v.end(), newData);
		labelsTrain.push_back(newData);
	}
	vector<float*> dataTrain;
	for (vector<float> v : dataTrainV) {
		float* newData = new float[v.size()];
		copy(v.begin(), v.end(), newData);
		dataTrain.push_back(newData);
	}
	cout << "Loaded data\n";
	
	// print out data, just to check
	/*for (int i = 50000; i < 50005; i++) {
		for (int j = 0; j < 10; j++) {
			cout << labelsTrain[i][j] << " ";
		}
		cout << ": ";
		for (int j = 0; j < 28*28; j++) {
			cout << dataTrain[i][j] << " ";
		}
		cout << endl << endl;
	}*/
	
	// parameters
	float eta = 0.001;
	int epochs = 1;
	int testSize = 10;
	float temperature = 0;
	
	// create model
	ModelOrganizer<float> model;
	model.addLayer(new FullyConnectedLayer<float>(28*28, 128));
	model.addLayer(new FullyConnectedLayer<float>(128, 10));
	//model.addLayer(new FullyConnectedLayer<float>(16, 10));
	
	// train model
	float error;
	for (int iter = 0; iter < epochs; iter++) {
		for (int i = 0; i < trainSize; i++) {
			error = model.trainCategoricalCrossEntropy(dataTrain[i], labelsTrain[i], eta, temperature);
			if (i % (10000) == 0) {
				cout << iter << "-" << i << ": " << error << endl;
			}
		}
	}
	
	//model.printLayerWeights();
	cout << "\n\nFinal Training Error: " << error << endl;
	cout << "Computing average training error...\n";
	
	// compute error
	float errorSum = 0;
	for (int i = 0; i < trainSize; i++) {
		errorSum += model.findErrorMax(dataTrain[i], labelsTrain[i]);
		if (i % (10000) == 0) {
			cout << i << ": " << (errorSum / i) << endl;
		}
	}
	cout << "\n\nFinal Average Training Error: " << (errorSum / (float)trainSize) << endl;
	
	// test
	model.addLayer(new MaxLayer<float>(10));
	for (int i = 0; i < testSize; i++) {
		float* res = new float[10];
		res[0] = 100;
		model.runInference(dataTrain[i], res);
		for (int j = 0; j < 10; j++) {
			cout << labelsTrain[i][j] << " ";
		}
		cout << "-> ";
		for (int j = 0; j < 10; j++) {
			cout << res[j] << " ";
		}
		cout << "\n";
		delete[] res;
	}
	
}

void expTwo() {
	
	// hyper parameters
	int epochs = 1;
	float eta = 0.001;
	int printOutput = 1000000;
	int testSize = 10;
	
	// generate data
	int trainSize = 1000000;
	int inputSize = 5;
	int outputSize = 5;
	vector<float*> data(trainSize);
	vector<float*> labels(trainSize);
	for (int i = 0; i < trainSize; i++) {
		float* d = new float[inputSize];
		float* l = new float[outputSize];
		for (int j = 0; j < inputSize; j++) {
			d[j] = (float)rand() / RAND_MAX;
			l[j] = d[j];
		}
		data[i] = d;
		labels[i] = l;
	}
	cout << "Generated data\n";
	
	// setup model
	ModelOrganizer<float> model;
	model.addLayer(new FullyConnectedLayer<float>(inputSize, outputSize));
	model.addLayer(new SigmoidLayer<float>(outputSize));
	cout << "Generated model\n";
	
	// train
	float error;
	for (int iter = 0; iter < epochs; iter++) {
		for (int i = 0; i < trainSize; i++) {
			error = model.trainMSE(data[i], labels[i], eta);
			if (i % printOutput == 0) {
				cout << iter << ": " << i << " " << error << endl;
			}
		}
	}
	cout << "Final: " << error << endl;
	
	// test
	for (int i = 0; i < testSize; i++) {
		float* res = new float[outputSize];
		model.runInference(data[i], res);
		for (int j = 0; j < outputSize; j++) {
			cout << labels[i][j] << " ";
		}
		cout << " -> ";
		for (int j = 0; j < outputSize; j++) {
			cout << res[j] << " ";
		}
		cout << "\n";
	}
	
	// summary
	cout << endl;
	model.printLayerWeights();
	
}

void expThree() {
	
	// hyper parameters
	int epochs = 1;
	float eta = 0.001;
	int printOutput = 1000000;
	int testSize = 10;
	
	// generate data
	int trainSize = 10000000;
	int inputSize = 1;
	int outputSize = 1;
	vector<float*> data(trainSize);
	vector<float*> labels(trainSize);
	for (int i = 0; i < trainSize; i++) {
		float* d = new float[inputSize];
		float* l = new float[outputSize];
		for (int j = 0; j < inputSize; j++) {
			float index = (float) (j % 100) / 100.0 * 2 * 3.14;
			d[j] = index;
			l[j] = sin(index);
		}
		data[i] = d;
		labels[i] = l;
	}
	cout << "Generated data\n";
	
	// setup model
	ModelOrganizer<float> model;
	model.addLayer(new FullyConnectedLayer<float>(inputSize, 3));
	//model.addLayer(new FullyConnectedLayer<float>(3, 3));
	model.addLayer(new SigmoidLayer<float>(3));
	model.addLayer(new FullyConnectedLayer<float>(3, 1));
	cout << "Generated model\n";
	
	// train
	float error;
	for (int iter = 0; iter < epochs; iter++) {
		for (int i = 0; i < trainSize; i++) {
			error = model.trainMSE(data[i], labels[i], eta);
			if (i % printOutput == 0) {
				cout << iter << ": " << i << " " << error << endl;
			}
		}
	}
	cout << "Final: " << error << endl;
	
	// test
	for (int i = 0; i < testSize; i++) {
		float* res = new float[outputSize];
		model.runInference(data[i], res);
		for (int j = 0; j < outputSize; j++) {
			cout << labels[i][j] << " ";
		}
		cout << " -> ";
		for (int j = 0; j < outputSize; j++) {
			cout << res[j] << " ";
		}
		cout << "\n";
	}
	
	// summary
	cout << endl;
	model.printLayerWeights();
	
}

int main() {
	
	expThree();
	
	return 0;
}
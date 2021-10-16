#pragma once
#include "headers.h"

template<class T>
class ModelOrganizer {
public:
	void addLayer(Layer<T>* layer) {
		// TODO: check for consistency
		layers.push_back(layer);
	}
	
	void runInference(const T* in, T* out) {
		T* curr = new T[getInputSize()];
		std::copy(in, in + getInputSize(), curr);
		for (Layer<T>* layer : layers) {
			//std::cout << "start\n";
			T* next = new T[layer->getNumOutputs()];
			//std::cout << "inited next\n";
			layer->forward(curr, next);
			//std::cout << "ran forward\n";
			delete[] curr;
			//std::cout << "deleted curr\n";
			curr = next;
			//std::cout << "end\n";
		}
		std::copy(curr, curr + getOutputSize(), out);
		//out = curr;
		delete[] curr;
	}
	
	// same thing as runInference, but tells each layer to store its input (uses more memory, but preps for training)
	void runInferenceStore(const T* in, T* out) {
		T* curr = new T[getInputSize()];
		std::copy(in, in + getInputSize(), curr);
		for (Layer<T>* layer : layers) {
			//std::cout << "start\n";
			T* next = new T[layer->getNumOutputs()];
			//std::cout << "inited next\n";
			layer->forwardStore(curr, next);
			//std::cout << "ran forward\n";
			delete[] curr;
			//std::cout << "deleted curr\n";
			curr = next;
			//std::cout << "end\n";
		}
		std::copy(curr, curr + getOutputSize(), out);
		//out = curr;
		delete[] curr;
	}
	
	// run after 'runInferenceStore'
	void runBackAndUpdateWeights(const T* error) {
		T* currError = new T[getOutputSize()];
		std::copy(error, error + getOutputSize(), currError);
		for (int i = layers.size()-1; i >= 0; i--) {
			Layer<T>* layer = layers[i];
			T* nextError = new T[layer->getNumInputs()];
			layer->backAndUpdateWeightsStore(currError, nextError);
			delete[] currError;
			currError = nextError;
		}
		delete[] currError;
	}
	
	T findError(const T* in, const T* correctOut) {
		T* result = new T[getOutputSize()];
		runInference(in, result);
		
		/*std::cout << "Correct | Out\n";
		for (int i = 0; i < 10; i++) {
			std::cout << correctOut[i] << " | " << result[i] << "\n";
		}*/
		
		T errorRes = 0;
		for (int i = 0; i < getOutputSize(); i++) {
			T diff = correctOut[i] - result[i];
			errorRes += diff * diff;
		}
		
		delete[] result;
		return sqrt(errorRes) / (float) getOutputSize();
	}
	
	// returns zero if max of result is in same index as max of correct output, 1 otherwise
	T findErrorMax(const T* in, const T* correctOut) {
		T* result = new T[getOutputSize()];
		runInference(in, result);
		
		/*std::cout << "Correct | Out\n";
		for (int i = 0; i < 10; i++) {
			std::cout << correctOut[i] << " | " << result[i] << "\n";
		}*/
		
		T maxRes = result[0];
		T maxResInd = 0;
		T maxCorrect = correctOut[0];
		T maxCorrectInd = 0;
		for (int i = 0; i < getOutputSize(); i++) {
			if (result[i] > maxRes) {
				maxRes = result[i];
				maxResInd = i;
			}
			if (correctOut[i] > maxCorrect) {
				maxCorrect = correctOut[i];
				maxCorrectInd = i;
			}
		}
		
		delete[] result;
		return (maxResInd != maxCorrectInd);
	}
	
	T trainCategoricalCrossEntropy(const T* in, const T* correctOut, const T& eta, const T& temperature) { // softmax + cross entropy
		// run inference
		T* result = new T[getOutputSize()];
		runInferenceStore(in, result);
		
		// init for error
		T lossCE = 0;
		T* error = new T[getOutputSize()];
		
		// find c = max of result[i]
		T c = 0;
		for (int i = 0; i < getOutputSize(); i++) {
			if (result[i] > c) {
				c = result[i];
			}
		}
		if (c > 1e5) {
			std::cout << "C: " << c << std::endl;
		}
		
		// find y = ln(sum e^res[i])
		T y = 0;
		for (int i = 0; i < getOutputSize(); i++) {
			y += exp(result[i] - c);
			if (isnan(y) || isinf(y)) {
				std::cout << i << ": " << y << " " << result[i] << " " << c << std::endl;
			}
		}
		y = c + log(y);
		
		// calc error
		for (int i = 0; i < getOutputSize(); i++) {
			T d = exp(result[i] - y + temperature);
			if (isnan(d) || isinf(d)) {
				std::cout << d << " " << result[i] << " " << y << std::endl;
				exit(0);
			}
			if (correctOut[i] == 1) {
				error[i] = -eta * (d - 1);
				lossCE = -log(d);
			} else {
				error[i] = -eta * d;
			}
		}
		
		// run error backwards through network
		runBackAndUpdateWeights(error);
		
		// cleanup and return result
		delete[] result;
		delete[] error;
		return lossCE;
	}
	
	// returns the error
	T trainMSE(const T* in, const T* correctOut, const T& eta) {
		// run inference
		T* result = new T[getOutputSize()];
		runInferenceStore(in, result);
		
		// find the error
		T errorRes = 0;
		T* error = new T[getOutputSize()];
		for (int i = 0; i < getOutputSize(); i++) {
			T diff = correctOut[i] - result[i];
			error[i] = -eta * diff;
			errorRes += diff * diff;
		}
		
		// run error backwards through network
		runBackAndUpdateWeights(error);
		
		// cleanup and return result
		delete[] result;
		delete[] error;
		return sqrt(errorRes) / (float) getOutputSize();
	}
	
	int getMemoryTrain() { return -1; } // TODO
	int getMemoryInference() { return -1; } // TODO
	
	int getInputSize() {
		if (layers.size() == 0) {
			return -1;
		}
		return layers[0]->getNumInputs();
	}
	
	int getOutputSize() {
		if (layers.size() == 0) {
			return -1;
		}
		return layers[layers.size()-1]->getNumOutputs();
	}
	
	void printLayerWeights() {
		for (Layer<T>* layer : layers) {
			layer->printWeights();
		}
	}
	
private:
	std::vector<Layer<T>* > layers;
	
	
};
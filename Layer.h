/* Layer class
 * Models y_bar = Layer(x_bar, weights_bar)
 * y_bar is array of output, x_bar is array of input, weights_bar is array of weights
 * Provides several pure virtual functions:
 *   forward: given x_bar, computes y_bar
 *   forwardStore: given x_bar, computes y_bar, but also stores x_bar (must call clear to get rid of memory)
 *   back: given x_bar and error_bar, computes partial error_bar wrt each input x in x_bar
 *   backStore: given x_bar and error_bar, computes partial error_bar wrt each input x in x_bar (uses stored x_bar)
 *   updateWeights: given x_bar and error_bar, changes the value of each weight
 *   updateWeightsStore: given x_bar and error_bar, changes the value of each weight (uses stored x_bar)
 *   backAndUpdateWeights: given x_bar and error_bar, computes partial error_bar wrt each input x in x_bar & updates values of each weight
 *   backAndUpdateWeights: given x_bar and error_bar, computes partial error_bar wrt each input x in x_bar & updates values of each weight (uses stored x_bar)
 *   getMemoryTrain: returns number of T stored to train layer
 *   getMemoryInference: returns number of T stored to run inference
 *   clearStored: clears stored input (but not weights!)
 */

#pragma once
#include "headers.h"

template <class T>
class Layer {
public:
	int getNumInputs() { return numInputs; }
	int getNumOutputs() { return numOutputs; }
	int getNumWeights() { return numWeights; }
	virtual void forward(const T* in, T* out) = 0; // in (length numInputs) is x_bar; out (length numOutputs) is y_bar
	virtual void back(const T* in, const T* error, T* out) = 0; // in (length numInputs) is x_bar; error (length numOutputs) is error_bar; out (length numInputs) is y_bar wrt x_bar
	virtual void updateWeights(const T* in, const T* error) = 0; // in (length numInputs) is x_bar; error (length numOutputs) is error in y_bar
	
	virtual void backAndUpdateWeights(const T* in, const T* error, T* out) {
		back(in, error, out);
		updateWeights(in, error);
	}
	
	virtual void backAndUpdateWeightsStore(const T* error, T* out) {
		backStore(error, out);
		updateWeightsStore(error);
	}
	
	void forwardStore(const T* in, T* out) {
		storedInput = new T[getNumInputs()];
		std::copy(in, in + getNumInputs(), storedInput);
		forward(in, out);
	}
	
	void backStore(const T* error, T* out) {
		back(storedInput, error, out);
	}
	
	void updateWeightsStore(const T* error) {
		updateWeights(storedInput, error);
	}
	
	void clearStored() {
		if (storedInput != nullptr) {
			delete[] storedInput;
			storedInput = nullptr;
		}
	}
	
	virtual void printWeights() = 0;
protected:
	Layer(int numInputs, int numOutputs, int numWeights) : numInputs(numInputs), numOutputs(numOutputs), numWeights(numWeights) {
		weights = new T[numWeights];
	}
	~Layer() {
		delete[] weights;
		clearStored();
	}
	int numInputs;
	int numOutputs;
	int numWeights;
	T* weights;
	T* storedInput = nullptr;
};
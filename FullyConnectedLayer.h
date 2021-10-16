#pragma once
#include "headers.h"

template <class T>
class FullyConnectedLayer : public Layer<T> {
public:
	FullyConnectedLayer(int inputSize, int outputSize) : Layer<T>(inputSize, outputSize, inputSize * outputSize + outputSize) {
		initWeights();
	}
	
	~FullyConnectedLayer() {} // this automatically calls ~Layer()
	
	void forward(const T* in, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			out[i] = 0;
			for (int j = 0; j < this->getNumInputs() + 1; j++) {
				if (j != this->getNumInputs()) {
					out[i] += in[j] * this->weights[j * this->getNumOutputs() + i];
				} else {
					out[i] += 1 * this->weights[j * this->getNumOutputs() + i];
				}
			}
		}
	}
	
	void back(const T* in, const T* error, T* out) {
		for (int i = 0; i < this->getNumInputs(); i++) {
			out[i] = 0;
			for (int j = 0; j < this->getNumOutputs(); j++) {
				out[i] += error[j] * this->weights[j + this->getNumOutputs() * i];
			}
		}
	}
	
	void updateWeights(const T* in, const T* error) {
		for (int i = 0; i < this->getNumInputs() + 1; i++) {
			for (int j = 0; j < this->getNumOutputs(); j++) {
				if (i != this->getNumInputs()) {
					this->weights[i*this->getNumOutputs() + j] += in[i] * error[j];
				} else {
					this->weights[i*this->getNumOutputs() + j] += 1 * error[j];
				}
			}
		}
	}
	
	void printWeights() {
		std::cout << "Fully Connected Layer: In: " << this->getNumInputs() << " Out: " << this->getNumOutputs() << "\n";
		for (int i = 0; i < this->getNumInputs() + 1; i++) {
			for (int j = 0; j < this->getNumOutputs(); j++) {
				std::cout << this->weights[j + (this->getNumOutputs()) * i] << " ";
			}
			std::cout << "\n";
		}
	}
	
private:
	void initWeights() {
		for (int i = 0; i < this->getNumInputs() + 1; i++) {
			for (int j = 0; j < this->getNumOutputs(); j++) {
				//r = ((double) rand() / (RAND_MAX))
				this->weights[i*this->getNumOutputs() + j] = (float)rand() / (RAND_MAX);
			}
		}
	}
};
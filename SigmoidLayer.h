#include "headers.h"

template<class T>
class SigmoidLayer : public Layer<T> {
public:
	SigmoidLayer(int size) : Layer<T>(size, size, 0) {}
	
	~SigmoidLayer() {} // this automatically calls ~Layer()
	
	void forward(const T* in, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			out[i] = sigmoid(in[i]);
		}
	}
	
	void back(const T* in, const T* error, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			T s = sigmoid(in[i]);
			out[i] = error[i] * s * (1 - s);
			//std::cout << out[i] << " ";
		}
		//std::cout << "\n";
	}
	
	void updateWeights(const T* in, const T* error) {}
	
	void printWeights() {
		std::cout << "Sigmoid Layer: In: " << this->getNumInputs() << " Out: " << this->getNumOutputs() << "\n";
	}
	
private:
	T sigmoid(const T& in) {
		return 1 / (1 + exp(-in));
	}
};
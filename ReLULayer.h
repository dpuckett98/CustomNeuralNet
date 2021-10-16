#include "headers.h"

template<class T>
class ReLULayer : public Layer<T> {
public:
	ReLULayer(int size) : Layer<T>(size, size, 0) {}
	
	~ReLULayer() {} // this automatically calls ~Layer()
	
	void forward(const T* in, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			out[i] = max(0, in[i]);
		}
	}
	
	void back(const T* in, const T* error, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			out[i] = (in[i] > 0) * error[i];
		}
	}
	
	void updateWeights(const T* in, const T* error) {}
	
	void printWeights() {
		std::cout << "ReLU Layer: In: " << this->getNumInputs() << " Out: " << this->getNumOutputs() << "\n";
	}
	
private:
	T max(T a, T b) {
		if (a > b) {
			return a;
		}
		return b;
	}
};
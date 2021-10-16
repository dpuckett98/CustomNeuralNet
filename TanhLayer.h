#include "headers.h"

template<class T>
class TanhLayer : public Layer<T> {
public:
	TanhLayer(int size) : Layer<T>(size, size, 0) {}
	
	~TanhLayer() {} // this automatically calls ~Layer()
	
	void forward(const T* in, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			out[i] = tanh(in[i]);
		}
	}
	
	void back(const T* in, const T* error, T* out) {
		for (int i = 0; i < this->getNumOutputs(); i++) {
			T s = tanh(in[i]);
			out[i] = error[i] * (1 - s*s);
			//std::cout << out[i] << " ";
		}
		//std::cout << "\n";
	}
	
	void updateWeights(const T* in, const T* error) {}
	
	void printWeights() {
		std::cout << "Tanh Layer: In: " << this->getNumInputs() << " Out: " << this->getNumOutputs() << "\n";
	}
	
private:
	T tanh(const T& in) {
		return (2 / (1 + exp(-2*in))) - 1;
	}
};
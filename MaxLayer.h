#include "headers.h"

template<class T>
class MaxLayer : public Layer<T> {
public:
	MaxLayer(int size) : Layer<T>(size, size, 0) {}
	
	~MaxLayer() {} // this automatically calls ~Layer()
	
	void forward(const T* in, T* out) {
		int maxIndex = 0;
		T maxVal = in[0];
		for (int i = 0; i < this->getNumOutputs(); i++) {
			if (in[i] > maxVal) {
				maxIndex = i;
				maxVal = in[i];
			}
		}
		for (int i = 0; i < this->getNumOutputs(); i++) {
			if (i == maxIndex) {
				out[i] = 1;
			} else {
				out[i] = 0;
			}
		}
	}
	
	void back(const T* in, const T* error, T* out) {
		// not implemented!
	}
	
	void updateWeights(const T* in, const T* error) {}
	
	void printWeights() {
		std::cout << "Max Layer: In: " << this->getNumInputs() << " Out: " << this->getNumOutputs() << "\n";
	}
};
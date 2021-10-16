#include "headers.h"

template<class T>
class SoftMaxLayer : public Layer<T> {
public:
	SoftMaxLayer(int size) : Layer<T>(size, size, 0) {}
	
	~SoftMaxLayer() {} // this automatically calls ~Layer()
	
	void forward(const T* in, T* out) {
		T sum = softmaxSum(in);
		for (int i = 0; i < this->getNumOutputs(); i++) {
			out[i] = softmax(in[i], sum);
		}
	}
	
	void back(const T* in, const T* error, T* out) {
		T sum = softmaxSum(in);
		for (int i = 0; i < this->getNumInputs(); i++) {
			out[i] = 0;
		}
		for (int i = 0; i < this->getNumOutputs(); i++) { // output index
			for (int j = 0; j < this->getNumInputs(); j++) { // input index
				T smi = exp(in[i]) / sum;
				T smj = exp(in[j]) / sum;
				if (i == j) {
					out[j] += smi * (1 - smj);
				} else {
					out[j] += -smi * smj;
				}
			}
		}
	}
	
	void updateWeights(const T* in, const T* error) {}
	
	void printWeights() {
		std::cout << "Max Layer: In: " << this->getNumInputs() << " Out: " << this->getNumOutputs() << "\n";
	}
private:
	T softmax(const T& val, const T& sum) {
		return exp(val) / sum;
	}
	
	T softmaxSum(const T* in) {
		T sum = 0;
		for (int i = 0; i < this->getNumOutputs(); i++) {
			sum += exp(in[i]);
		}
		return sum;
	}
};
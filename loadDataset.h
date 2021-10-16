#include <fstream>
#include <string>
#include <sstream>
#include <vector>

int loadDatasetMNIST(std::string mnistTrainLoc, std::vector<std::vector<float> >& labelsTrain, std::vector<std::vector<float> >& dataTrain) {
	// open file
	std::ifstream file(mnistTrainLoc);
	// init vars
	const int trainSize = 60000;
	const int numClasses = 10;
	std::vector<int> labels = std::vector<int>(trainSize);
	dataTrain = std::vector<std::vector<float> >(trainSize);
	labelsTrain = std::vector<std::vector<float> >(trainSize);
	// load data from file
	for (int i = 0; i < trainSize; i++) {
		// read in current line
		std::string line;
		getline(file, line, '\n');
		std::stringstream s(line);
		std::string val;
		getline(s, val, ',');
		labels[i] = stoi(val);
		std::vector<float> curr(28*28);
		for (int j = 0; j < 28*28; j++) {
			getline(s, val, ',');
			curr[j] = (float)stoi(val) / (float)255;
		}
		dataTrain[i] = curr;
	}
	// one-hot encode labels
	for (int i = 0; i < trainSize; i++) {
		//float* res = new float[numClasses];
		std::vector<float> res(numClasses);
		for (int j = 0; j < numClasses; j++) {
			if (j == labels[i]) {
				res[j] = 1;
			} else {
				res[j] = 0;
			}
		}
		labelsTrain[i] = res;
	}
	return trainSize;
}
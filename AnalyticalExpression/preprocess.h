// CNNs with Label Noise - code for the paper "The Resistance to Label Noise in K-NN and CNN Depends on its Concentration" by Amnon Drory, Oria Ratzon, Shai Avidan and Raja Giryes
// 
// MIT License
// 
// Copyright (c) 2019 Amnon Drory
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef __CWN_PREPROCESS_H
#define __CWN_PREPROCESS_H

#include <string>
using std::string;
#include <vector>
using std::vector;
#include "CWN_arr.h"

class CPreProcessor
{
public:
	CPreProcessor() {}
	void go(int argc, char *argv[]);
	int num_classes() const { return m_num_classes; }
	int num_samples() const { return m_num_samples; }
	
	CWN_arr q_clean; 
	CWN_arr q_noisy; 	
private:
	vector<string> SplitFileName(string filename);
	int getdir(string dir, vector<string> &files);
	void load_data(int argc, char *argv[]);
	void ReorderDistributions();
	
	CWN_arr GT_Labels; 
	CWN_arr ConfMat;
	int m_num_samples;
	int m_num_classes; 
};
#endif // __CWN_PREPROCESS_H

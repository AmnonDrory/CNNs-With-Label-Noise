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

#include "preprocess.h"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>
using std::cout;
using std::endl;
#include <cstdlib>
using std::stoi;


static const string GT_LABELS_PREFIX = "Labels1Hot";
static const string CONF_PREFIX = "ConfusionMatrix";
static const string CLEAN_DIST_FILENAME = "Prediction.best.test.bin";

int CPreProcessor::getdir(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
		string s(dirp->d_name);
		if (s.length() > 2)
		{
			files.push_back(s);
		}
    }
    closedir(dp);
    return 0;
}

vector<string> CPreProcessor::SplitFileName(string filename)
{
	const int NUM_SUBSTRINGS=4;
	size_t starts[NUM_SUBSTRINGS];
	size_t ends[NUM_SUBSTRINGS];
	starts[0] = 0;
	ends[0] = filename.find('.')-1;
	for ( int i=1; i < NUM_SUBSTRINGS; i++)
	{
		starts[i] = ends[i-1]+2;
		if (i == (NUM_SUBSTRINGS-1))
		{
			ends[i] = filename.length()-1;
		}
		else
		{
			ends[i] = filename.find('.',starts[i])-1;
		}
	}

	vector<string> substrings;
	for (int i=0; i < NUM_SUBSTRINGS; i++)
	{
		size_t len = ends[i]-starts[i]+1;
		substrings.push_back(filename.substr(starts[i], len));
	}
	return substrings;
}

void CPreProcessor::load_data(int argc, char *argv[])
{
	if (argc==1)
	{
		cout << "Error: data dir must be provided as command line argument" << endl;
		exit(1);
	}
	string DataDir(argv[1]);
	if (DataDir.back() != '/') { DataDir += '/'; }
	vector<string> FilesList;
	getdir(DataDir,FilesList);
	vector<vector<string>> substrings;

	for (size_t i=0; i < FilesList.size(); i++)
	{
		string& filename=FilesList[i];
		substrings.push_back(SplitFileName(filename));
		if (substrings[i][0] == GT_LABELS_PREFIX)
		{				
			m_num_samples = stoi(substrings[i][1]);
			m_num_classes = stoi(substrings[i][2]);
		}
	}
	
	for (size_t i=0; i < FilesList.size(); i++)
	{
		string& filename=FilesList[i];
		if (filename == CLEAN_DIST_FILENAME)
		{
			q_clean.init(DataDir + filename, m_num_samples, m_num_classes, "float32");
		}
		else if (substrings[i][0] == CONF_PREFIX)
		{
			ConfMat.init(DataDir + filename, m_num_classes, m_num_classes, "float32");
		}
		else if (substrings[i][0] == GT_LABELS_PREFIX)
		{
			GT_Labels.init(DataDir + filename, m_num_samples, m_num_classes, "uint8");
		}
	}
}

static void swap(CWN_arr_class& a, CWN_arr_class& b)
{
	CWN_arr_class tmp=a;
	a=b;
	b=tmp;
}

void CPreProcessor::ReorderDistributions()
{
	for (int i = 0; i < q_clean.H(); i++)
	{
		int GT_ind = 0;
		for (int j=0; j < GT_Labels.W(); j++)
		{
			if (GT_Labels(i,j) == 1)
			{
				GT_ind = j;
				break;
			}
		}
			
		swap(q_clean(i,0), q_clean(i,GT_ind));
		swap(q_noisy(i,0), q_noisy(i,GT_ind));
	}
}

void CPreProcessor::go(int argc, char *argv[])
{
	load_data(argc, argv);
	if (argc > 2)
	{
		int sample_ind = stoi(argv[2]);
		if ((sample_ind >= m_num_samples) || (sample_ind < 0))
		{
			cout << "Invalid sample number. num_samples= " << m_num_samples << endl;
			exit(1);
		}
		q_clean.RemoveAllRowsExcept(sample_ind);
		GT_Labels.RemoveAllRowsExcept(sample_ind);
		m_num_samples = 1;
	}
	q_clean.TransformFromLogitsToSoftmax();
	MatrixMultiplication(q_clean, ConfMat, /*Output: */ q_noisy);
	ReorderDistributions();
}

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

// Compile with:
// 	g++ -std=c++11 1.cpp
#include "CWN_arr.h"
#include <iostream>
#include <fstream>
#include <algorithm> 
using std::cout ;
using std::endl ;
using std::ios;
using std::ifstream;
using std::min;
#include <cmath>
using std::exp;

void CWN_arr::TransformFromLogitsToSoftmax()
{
	double *row = new double[m_W]; // perform intermediate calculations in double accuracy
	for (int i=0; i < m_H; i++)
	{
		double exp_sum=0.0;
		CWN_arr_class row_max=at(i,0);
		for (int j=1; j<m_W; j++)
		{
			if (row_max < at(i,j)) {row_max = at(i,j);}
		}
		for (int j=0; j<m_W; j++)
		{
			row[j] = exp(at(i,j)-row_max);
			exp_sum += row[j];
		}
		for (int j=0; j<m_W; j++)
		{
			at(i,j) = static_cast<CWN_arr_class>(row[j] /= exp_sum);
		}		
	}
	delete[] row;
}

CWN_arr::CWN_arr() : data(NULL) {}

void CWN_arr::allocate(SIZE_T H, SIZE_T W)
{
	m_H = H;
	m_W = W;
	data = new CWN_arr_class[m_H*m_W];
}

void CWN_arr::init(const string& filename, SIZE_T H, SIZE_T W, const string& dtype)
{
	allocate(H,W);
	ifstream myFile (filename, ios::in | ios::binary);
	
	if (dtype == "uint8")
	{
		char* buffer = new char[m_W*m_H];
		myFile.read(buffer,m_H*m_W);
		for (int i = 0; i < m_H*m_W; i++)
		{
			data[i] = static_cast<CWN_arr_class>(buffer[i]);
		}
		delete[] buffer;
	}
	else if (dtype == "float32")
	{
		float* buffer = new float[m_W*m_H];
		myFile.read((char*)buffer,m_H*m_W*sizeof(float));
		for (int i = 0; i < m_H*m_W; i++)
		{
			data[i] = static_cast<CWN_arr_class>(buffer[i]);
		}
		delete[] buffer;
	}
	else
	{
		cout << "Error: unknown dtype " << dtype << endl;
		exit(1);
	}
}

CWN_arr::~CWN_arr()
{
	delete data; 
}

void CWN_arr::print(int max_rows)
{
	if ((max_rows < 0) || (max_rows > m_H))	{ max_rows = m_H; }
	
	for (int i=0; i<max_rows; i++)
	{
		for (int j=0; j < m_W; j++)
		{
			cout << at(i,j) << " ";
		}
		cout << endl;
	}
}

void CWN_arr::RemoveAllRowsExcept(int row_ind)
{
	CWN_arr_class* new_data = new CWN_arr_class[m_W];
	for (int j=0; j < m_W; j++)
	{
		new_data[j] = at(row_ind,j);
	}
	delete[] data;
	data = new_data;
	m_H = 1;
}


void MatrixMultiplication(const CWN_arr& A, const CWN_arr& B, CWN_arr& Result)
{
	if (A.W() != B.H())
	{
		cout << "Error: MatrixMultiplication received incompatible sizes [" << A.H() << "," << A.W() << "] [" << B.H() << "," << B.W() << "]" << endl;
		exit(1);
	}
	
	Result.allocate(A.H(), B.W());
	for (int i = 0; i < A.H(); i++)
	{
		for (int j = 0; j < B.W(); j++)
		{
			double sum=0.0;
			for (int k = 0; k < B.H(); k++)
			{
				sum += static_cast<double>(A(i,k)) * static_cast<double>(B(k,j));
			}
			Result(i,j) = static_cast<CWN_arr_class>(sum);
		}
	}
}

void ConvexCombination(mpf_class Gamma, const CWN_arr& A, const CWN_arr& B, /*output: */ CMpfArr2D& C)
{
	if ( (A.W()!=B.W()) || (A.H()!=B.H()) || (C.W()!=B.W()) || (C.H()!=B.H()) )
	{
		cout << "Error: inconsistent sizes in ConvexCombination" << endl;
		exit(1);
	}
	mpf_class OneMinusGamma = (1.0 - Gamma);

	int num_cells = A.H()*A.W();
	for (int i=0; i < num_cells; i++)
	{
		C.data[i] = OneMinusGamma*A.data[i] + Gamma*B.data[i];
	}
}

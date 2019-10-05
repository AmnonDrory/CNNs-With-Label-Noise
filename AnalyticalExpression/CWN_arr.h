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

#ifndef __CWN_ARR_H
#define __CWN_ARR_H

#include <string>
#include <vector>
using std::string; 
using std::vector;
#include <gmpxx.h> 
#define SIZE_T int

class CWN_arr;

class CMpfArr2D
{
public:
	CMpfArr2D() : data(NULL) {}
	
	CMpfArr2D(int H, int W)
	{
		init(H,W);
	}
	
	void init(int H, int W)
	{
		data = new mpf_class[H*W];
		m_H=H;
		m_W=W;
	}
	
	~CMpfArr2D()
	{
		delete[] data;
	}
	
	mpf_class& at(int row, int col)
	{
		return data[row*m_W + col];
	}
	
	const mpf_class& at(int row, int col) const
	{
		return data[row*m_W + col];
	}	
	
	mpf_class& operator()(int row, int col)
	{
		return at(row, col);
	}
	
	const mpf_class& operator()(int row, int col) const
	{
		return at(row, col);
	}	
	
	int H() const { return m_H; }
	int W() const { return m_W; }
	
	friend void ConvexCombination(mpf_class Gamma, const CWN_arr& A, const CWN_arr& B, /*output: */ CMpfArr2D& combined_q);
		
private:
	mpf_class *data;
	int m_H;
	int m_W;
};

class CMpfArr3D
{
public:
	CMpfArr3D() : data(NULL) {}
	
	CMpfArr3D(int D, int H, int W)
	{
		init(D,H,W);
	}
	
	void init(int D, int H, int W)
	{
		data = new mpf_class[D*H*W];
		m_H=H;
		m_W=W;
		m_D=D;		
	}
	
	~CMpfArr3D()
	{
		delete[] data;
	}
	
	mpf_class& at(int layer, int row, int col)
	{
		return data[(layer*m_H + row)*m_W + col];
	}
	
	mpf_class& operator()(int layer, int row, int col)
	{
		return at(layer, row, col);
	}
		
private:
	mpf_class *data;
	int m_D;	
	int m_H;
	int m_W;
};

typedef double CWN_arr_class;

class CWN_arr
{
private:
	CWN_arr_class* data;
	SIZE_T m_W;
	SIZE_T m_H;
	int sub2ind(int i, int j) const;
	void allocate(SIZE_T H, SIZE_T W);

public:
	CWN_arr();
	void init(const string& filename, SIZE_T H, SIZE_T W, const string& dtype);
	void RemoveAllRowsExcept(int row_ind);
	void TransformFromLogitsToSoftmax();
	SIZE_T W() const {return m_W;}
	SIZE_T H() const {return m_H;}
	~CWN_arr();
	void print(int max_rows=-1);
	CWN_arr_class& at(int row, int col);
	CWN_arr_class at(int row, int col) const;
	CWN_arr_class& operator()(int row, int col) { return at(row,col); }
	CWN_arr_class operator()(int row, int col) const { return at(row,col); }
	
	friend void MatrixMultiplication(const CWN_arr& A, const CWN_arr& B, CWN_arr& Result);	
	friend void ConvexCombination(mpf_class Gamma, const CWN_arr& A, const CWN_arr& B, /*output: */ CMpfArr2D& combined_q);
}; 

inline int CWN_arr::sub2ind(int row, int col) const
{
	return row*m_W + col;
}

inline CWN_arr_class& CWN_arr::at(int row, int col)
{
	return data[sub2ind(row,col)];
}

inline CWN_arr_class CWN_arr::at(int row, int col) const
{
	return data[sub2ind(row,col)];
}

void MatrixMultiplication(const CWN_arr& A, const CWN_arr& B, CWN_arr& Result);

void ConvexCombination(mpf_class Gamma, const CWN_arr& A, const CWN_arr& B, /*output: */ CMpfArr2D& C);

#endif // __CWN_ARR_H

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

// Was successfully compiled on Linux (Ubuntu 16.04) using the command line:
// 	g++ -std=c++11 -mcmodel=large -Ofast -Wall -pthread CWN_Analytical.cpp preprocess.cpp CWN_arr.cpp -o CWN_Analytical.out -lgmpxx -lgmp
//
// To Run:
// 	./CWN_Analytical.out ../predict/test/4385/ 101
//
// Command line arguments:
// 1: Directory containing datafiles created by running CnnWithNoise.py
// 2: optional: index of test sample for which the analytical graph will be created
//             (if not supplied, will create graph that is averaged over all test samples).
//
// The result is written into the file CWN_Analytical_result.txt.

#include <iostream> 
#include <cmath> 
#include <algorithm> 
#include <pthread.h> 
#include <cassert> 
#include <unistd.h> 
#include <ctime> 
#include <fstream> 
#include <iomanip>
#include <string>
#include <sstream>
using std::ofstream; 
using std::ifstream; 
using std::cout ; 
using std::endl ; 
using std::min; 
using std::max;
using std::flush;
using std::setw;
using std::string;
using std::stringstream;

#include "CWN_arr.h"
#include "preprocess.h"

static const string result_file_base = "CWN_Analytical_result";
static const string result_file_ext = ".txt";
static const int INVALID = -7;

static const int MAX_WORKER_THREADS = sysconf(_SC_NPROCESSORS_ONLN)-1; // one core is reserved for the main thread, the rest are for worker threads.
static const int MAX_K = 80; // For a quick small-scale test - make this smaller
static const int K_STEP = 1; 
static const int MIN_K = 1;
static const double GAMMA_VALS[] = {0.0, 0.15, 0.3, 0.45, 0.6, 0.75}; 
static const int NUM_GAMMA_VALS = sizeof(GAMMA_VALS)/sizeof(GAMMA_VALS[0]);

static const int MAX_N = 2*(1+MAX_K); // This is a coarse upper bound

CPreProcessor P;
CMpfArr2D binom(MAX_N,MAX_N); // binomial coefficients 
CMpfArr2D q_powers; // q_powers(j,p) equals probability for class j, raised to the power of p
CMpfArr3D intermediate_arr; // A cache of intermediate results that are shared by many caclculations

//------------------------------------------------------------------(o)

void pre_calculate_binomial_coefficients()
{
	for (int n = 0; n < MAX_N; n++)
	{
		binom(n,0) = 1.0;
	}
	
	for (int k=1; k < MAX_N; k++)
	{
		for (int n = k; n < MAX_N; n++)
		{
			mpf_class ratio = (mpf_class(n)/mpf_class(k));
			binom(n,k) = ratio*binom(n-1,k-1);
		}
	}
}

//------------------------------------------------------------------(o)
typedef struct {
	int num;	
	int remaining;
	int max_other; 
	int most_taken_by_later;
	mpf_class sum;
} inner_loop_args_t; 

void* inner_loop(void* args)
{
	int remaining = (*((inner_loop_args_t*)args)).remaining;
	int max_other = (*((inner_loop_args_t*)args)).max_other;
	int most_taken_by_later = (*((inner_loop_args_t*)args)).most_taken_by_later;
	
	
	int* num = new int[P.num_classes()];
	int* top = new int[P.num_classes()];
	int* bottom = new int[P.num_classes()];
	mpf_class *sums = new mpf_class[P.num_classes()+1];
	
	for (int i = 0; i < P.num_classes(); i++)
	{
		num[i] = INVALID;
		top[i] = INVALID;
		bottom[i] = INVALID;
		sums[i] = INVALID;
	}
	sums[P.num_classes()] = INVALID;
	
	enum recursion_stage_t {PRE, POST} mode = PRE;
	int cur_ind = 1; // cur_ind = which label we are now handling	

	while (true)
	{
		if (mode == PRE)
		{
			if (cur_ind == P.num_classes()) // one after last label
			{
				mode = POST;
				sums[cur_ind]=1.0;
				cur_ind -= 1;
				continue;
			}

			assert(remaining < MAX_N);
			assert(remaining >= 0);
			assert(max_other < MAX_N);
			assert(max_other >= 0);
			assert(cur_ind < P.num_classes());

			if ((num[cur_ind] == INVALID) && (intermediate_arr(max_other,cur_ind,remaining) != INVALID))
			{
				mode = POST;
				sums[cur_ind]=intermediate_arr(max_other,cur_ind,remaining);
				cur_ind -= 1;
				continue;
			}			
			
			// 1. Calculate top and bottom:
			if (top[cur_ind] == INVALID)
			{
				most_taken_by_later -= max_other; // remove current label from the count
				top[cur_ind] = min(remaining, max_other);
				bottom[cur_ind] = max(int(0), remaining - most_taken_by_later);
			}
			
			// 2. select number
			if (num[cur_ind] == INVALID)
			{
				num[cur_ind] = bottom[cur_ind];
				remaining -= num[cur_ind];
				sums[cur_ind] = 0;
			}
			else
			{
				num[cur_ind] += 1;
				remaining -= 1;
			}
			
			// 3. move inwards
			cur_ind += 1;
		}
		
		else if (mode == POST)
		{
			
			if (cur_ind == 0)  // first label
			{
				break;
			}
			
			mpf_class coeff = binom(remaining + num[cur_ind], num[cur_ind]) * q_powers(cur_ind, num[cur_ind]);
			mpf_class addition = coeff * sums[cur_ind+1];
			sums[cur_ind] += addition;
			
			if (num[cur_ind] == top[cur_ind])
			{
				// clean up and backtrack to previous index
				remaining += num[cur_ind];
				intermediate_arr(max_other,cur_ind,remaining) = sums[cur_ind];
				most_taken_by_later += max_other;
				num[cur_ind] = INVALID;
				top[cur_ind] = INVALID;
				bottom[cur_ind] = INVALID;
				cur_ind -= 1; // move outward
			}
			else
			{
				mode = PRE; // process next value for current label			
			}
		}
	}
	
	(*((inner_loop_args_t*)args)).sum = sums[1];
	
	delete[] num;
	delete[] top;
	delete[] bottom;
	delete[] sums;
	
	return NULL;
}

//------------------------------------------------------------------(o)
void calculate_q_powers(int sample_ind, mpf_class Gamma)
{
	mpf_class OneMinusGamma = (1.0 - Gamma);
	for (int class_ind = 0; class_ind < P.num_classes(); class_ind++)
	{
		q_powers(class_ind,0) = 1.0;
		q_powers(class_ind,1) = ( 	OneMinusGamma*P.q_clean(sample_ind,class_ind) +
									Gamma * P.q_noisy(sample_ind,class_ind));
	}
	
	for (int power=2; power < MAX_N; power++)
	{
		for (int class_ind = 0; class_ind < P.num_classes(); class_ind++)
		{
			q_powers(class_ind,power) = q_powers(class_ind,1) * q_powers(class_ind,power-1);
		}
	}
}

//------------------------------------------------------------------(o)
mpf_class probability_of_plurality(int K, int advantage=1)
{
	pthread_t threads[MAX_WORKER_THREADS];
	inner_loop_args_t inner_args[MAX_WORKER_THREADS];
	bool thread_running[MAX_WORKER_THREADS]={0};
	int next_thread_ind=0;
	
	for (int i = 0; i < MAX_WORKER_THREADS; i++ )
	{ thread_running[i] = false; }	
		
	int top = K;
	double bottom_raw = ceil(double(K + ((P.num_classes() - 1) * advantage)) / double(P.num_classes()));
	int bottom = max(0, int(bottom_raw));
	
	int most_taken_by_later = INVALID;
	int max_other = INVALID;

	mpf_class sum = 0.0;

	for (int num = bottom; num <= top; num++)
	{
		// find next thread that has finished 
		while(true)
		{
			next_thread_ind += 1;
			if (next_thread_ind >= MAX_WORKER_THREADS)
				{ next_thread_ind = 0; }
			if (thread_running[next_thread_ind] == false)
			{
				break;
			}
			int thread_still_running = pthread_tryjoin_np(threads[next_thread_ind], NULL);
			if(thread_still_running == false)
			{
				break;
			}
		}
		
		if (thread_running[next_thread_ind])
		{
			int n_0 = inner_args[next_thread_ind].num;
			mpf_class coeff = binom(K,n_0) * q_powers(0,n_0);
			sum += coeff * inner_args[next_thread_ind].sum;
			thread_running[next_thread_ind] = false;
		}

		int remaining = K - num;
		max_other = max(0, num - advantage);  // the largest number that can be given to any label that isn''t 1
		most_taken_by_later = (P.num_classes() - 1) * max_other; // the most that can be given to all remaining unassigned labels
		
		inner_args[next_thread_ind].num = num;
		inner_args[next_thread_ind].remaining = remaining;
		inner_args[next_thread_ind].max_other = max_other;
		inner_args[next_thread_ind].most_taken_by_later = most_taken_by_later;
		pthread_create(&(threads[next_thread_ind]), NULL, &inner_loop, (void*)(&(inner_args[next_thread_ind])));
		thread_running[next_thread_ind] = true;
	}
	
	// Read out results from the last batch of running threads
	for (next_thread_ind = 0 ; next_thread_ind < MAX_WORKER_THREADS; next_thread_ind++)
	{
		if (thread_running[next_thread_ind])
		{
			pthread_join(threads[next_thread_ind], NULL);
			int n_0 = inner_args[next_thread_ind].num;
			mpf_class coeff = binom(K,n_0) * q_powers(0,n_0);
			sum += coeff * inner_args[next_thread_ind].sum;
			thread_running[next_thread_ind] = false;
		}		
	}
		
	return sum;
}

//------------------------------------------------------------------(o)
void initialize_intermediate_array()
{
	// Initialize intermediate array to invalid
	for (int ii = 0; ii < MAX_N; ii++)
	{
		for (int jj = 0; jj < P.num_classes(); jj++)
		{
			for (int kk = 0; kk < MAX_N; kk++)	
			{ 
				intermediate_arr(ii,jj,kk) = INVALID; 
			}		
		}
	}
}

//------------------------------------------------------------------(o)
void expected_accuracy(mpf_class Gamma, ofstream& result_file)
{
	const int NUM_K_VALS = (MAX_K - MIN_K)/K_STEP + 1;
	mpf_class sum[NUM_K_VALS];
	
	int k_ind=0;
	for (int K = MIN_K; K <= MAX_K; K+=K_STEP)
	{
		sum[k_ind] = 0.0;
		k_ind++;
	}
	
	for (int sample_ind = 0; sample_ind < P.num_samples(); sample_ind++)
	{
		initialize_intermediate_array();
		calculate_q_powers(sample_ind, Gamma);
		k_ind=0;
		for (int K = MIN_K; K <= MAX_K; K+=K_STEP)
		{
			sum[k_ind] += probability_of_plurality(K);
			k_ind++;
		}
	}
	
	k_ind = 0;
	for (int K = MIN_K; K <= MAX_K; K+=K_STEP)
	{
		mpf_class accuracy = sum[k_ind]/P.num_samples();
		result_file << ",\t";
		result_file << accuracy;
		k_ind++;
	}
	result_file << endl ;
}

//------------------------------------------------------------------(o)
int main(int argc, char *argv[])
{	
	bool Silent=false;
	if (argc > 3)
	{
		if (argv[3][0] == '1')
		{
			Silent = true;
		}		
	}
	P.go(argc,argv);
	q_powers.init(P.num_classes(),MAX_N); // q_powers(j,p) equals probability for class j, raised to the power of p

	intermediate_arr.init(MAX_N,P.num_classes(),MAX_N); // A cache of intermediate results that are shared by many caclculations
	
	time_t tstart, tend; 
	
	if (!Silent)
	{	cout << "MAX_WORKER_THREADS = " << MAX_WORKER_THREADS << endl; }
	
	static const string result_file_base = "CWN_Analytical_result";
	static const string result_file_ext = ".txt";
	string result_file_name;
	if (argc > 2)
	{
		result_file_name = result_file_base + "." + argv[2] + result_file_ext;
	}
	else
	{
		result_file_name = result_file_base + result_file_ext;
	}
		
	ofstream result_file(result_file_name);
	
	result_file << -1 << "     ";
	for (int K = MIN_K; K <= MAX_K; K+=K_STEP)
	{
		result_file << ",\t" << K ;
	}
	result_file << endl ;	
	
	pre_calculate_binomial_coefficients();	
	
	for (int gamma_ind = 0; gamma_ind < NUM_GAMMA_VALS; gamma_ind++)
	{
		mpf_class Gamma = GAMMA_VALS[gamma_ind];
		
		if (!Silent)
		{ cout << "Processing Gamma=" << Gamma << "      : " << flush ; }
		result_file << Gamma << "     ";
		tstart = time(0);			
		
		expected_accuracy(Gamma, result_file);
		
		tend = time(0); 
		if (!Silent)		
		{ cout << difftime(tend, tstart) <<" second(s)."<< endl;}
	}

	return 0;
}

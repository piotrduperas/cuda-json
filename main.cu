#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sstream>

using namespace std;

struct braces_to_numbers
{
  __host__ __device__
  char operator()(const char& x) const { 
    if(x == '{') return 1;
    if(x == '}') return -1;
    return 0;
  }
};

struct is_brace
{
  __host__ __device__
  bool operator()(const thrust::tuple<char, int>& x)
  {
    return x.get<0>() == '{' || x.get<0>() == '}';
  }
};

int main()
{
  cout << "CUDA JSON Validator" << endl << endl;


  ifstream ifile("github_events.json");

  ostringstream ss;
  ss << ifile.rdbuf();
  const string& s = ss.str();

  thrust::host_vector<char> H_file(s.begin(), s.end());
  size_t file_size = H_file.size();

  thrust::device_vector<char> D_file(file_size);
  thrust::device_vector<thrust::tuple<char, int>> D_braces_pos = thrust::device_vector<thrust::tuple<char, int>>(file_size);

  thrust::copy(H_file.begin(), H_file.end(), D_file.begin());

  auto char_and_pos = thrust::make_zip_iterator(thrust::make_tuple(D_file.begin(), thrust::make_counting_iterator(0)));
  
  auto last_brace_it = thrust::copy_if(char_and_pos, char_and_pos + s.length(), D_braces_pos.begin(), is_brace());

  for(auto it = D_braces_pos.begin(); it < last_brace_it; it++){
    cout << thrust::get<1>((thrust::tuple<char, int>)*it) << " - " << thrust::get<0>((thrust::tuple<char, int>)*it) << endl;
  }
  

  return 0;
}
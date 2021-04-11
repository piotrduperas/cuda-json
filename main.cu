#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sstream>

using namespace std;

typedef thrust::tuple<char, int> char_and_position;

struct braces_to_numbers
{
  __host__ __device__
  char_and_position operator()(const char_and_position& x) const { 
    if(x.get<0>() == '{') return char_and_position(1, x.get<1>());
    if(x.get<0>() == '}') return char_and_position(-1, x.get<1>());
    return char_and_position(0, x.get<1>());
  }
};

struct is_brace
{
  __host__ __device__
  bool operator()(const char_and_position& x)
  {
    return x.get<0>() == '{' || x.get<0>() == '}';
  }
};

struct get_char_from_tuple
{
  __host__ __device__
  char operator()(const char_and_position& x) const { 
    return x.get<0>();
  }
};

struct increment
{
  __host__ __device__
  short operator()(const short& x) const { 
    return x + 1;
  }
};

struct is_closing_brace
{
  __host__ __device__
  bool operator()(const char_and_position& x) const { 
    return x.get<0>() == -1;
  }
};

int main()
{
  cout << "CUDA JSON Validator" << endl << endl;


  ifstream ifile("senators_short.json");

  ostringstream ss;
  ss << ifile.rdbuf();
  const string& s = ss.str();

  thrust::host_vector<char> H_file(s.begin(), s.end());
  size_t file_size = H_file.size();

  thrust::device_vector<char> D_file(file_size);
  thrust::device_vector<char_and_position> D_braces_pos(file_size);

  thrust::copy(H_file.begin(), H_file.end(), D_file.begin());

  auto char_and_pos = thrust::make_zip_iterator(thrust::make_tuple(D_file.begin(), thrust::make_counting_iterator(0)));
  
  auto last_brace_it = thrust::copy_if(char_and_pos, char_and_pos + s.length(), D_braces_pos.begin(), is_brace());
  D_braces_pos.resize(last_brace_it - D_braces_pos.begin());


  thrust::transform(D_braces_pos.begin(), D_braces_pos.end(), D_braces_pos.begin(), braces_to_numbers());

  thrust::device_vector<short> D_levels(D_braces_pos.size());

  thrust::transform_inclusive_scan(D_braces_pos.begin(), D_braces_pos.end(), D_levels.begin(), get_char_from_tuple(), thrust::plus<short>());
  thrust::transform_if(D_levels.begin(), D_levels.end(), D_braces_pos.begin(), D_levels.begin(), increment(), is_closing_brace());

  for(int i = 0; i < D_braces_pos.size(); i++){
    cout << thrust::get<1>((char_and_position)D_braces_pos[i]) << " - " << (thrust::get<0>((char_and_position)D_braces_pos[i]) == 1 ? '{' : '}') << " - " << D_levels[i] << endl;
  }
  

  return 0;
}
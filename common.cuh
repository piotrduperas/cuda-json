#ifndef CUDA_JSON_COMMON_CUH
#define CUDA_JSON_COMMON_CUH

#include <thrust/tuple.h>
#include <iostream>
#include <string>
#include <chrono>

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

void elapsedTime(chrono::time_point<chrono::steady_clock> start, string description);

#endif
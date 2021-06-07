#ifndef CUDA_JSON_COMMON_CUH
#define CUDA_JSON_COMMON_CUH

#include <thrust/tuple.h>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

#define POSITIONS_TYPE char
#define LEVELS_TYPE char

typedef thrust::tuple<char, POSITIONS_TYPE> char_and_position;
typedef thrust::tuple<char, char> adjacent_chars;
typedef thrust::tuple<char, LEVELS_TYPE> char_and_level;
typedef thrust::tuple<char_and_level, char_and_level> adjacent_chars_and_levels;

struct braces_to_numbers
{
  __host__ __device__
  char operator()(const char& x) const { 
    switch(x) {
      case '{':
      case '[':
        return 1;
        break;
      default:
        return -1;
    }
  }
};

struct is_brace_or_bracket
{
  __host__ __device__
  bool operator()(const char_and_position& x)
  {
    char c = x.get<0>();
    return 
      c == '{' || 
      c == '}' ||
      c == '[' ||
      c == ']';
  }
};

struct increment
{
  __host__ __device__
  LEVELS_TYPE operator()(const LEVELS_TYPE& x) const { 
    return x + 1;
  }
};

struct is_closing_brace
{
  __host__ __device__
  bool operator()(const char& x) const { 
    return x == '}' || x == ']';
  }
};

struct is_brace
{
  __host__ __device__
  bool operator()(const char_and_level& x)
  {
    char c = x.get<0>();
    return 
      c == '{' || 
      c == '}';
  }
};

struct opening_and_closing_chars_have_the_same_level
{
  __host__ __device__
  bool operator()(const adjacent_chars_and_levels& x)
  {
    char_and_level cl1 = x.get<0>();
    char_and_level cl2 = x.get<1>();

    char c1 = cl1.get<0>();
    char c2 = cl2.get<0>();
    LEVELS_TYPE l1 = cl1.get<1>();
    LEVELS_TYPE l2 = cl2.get<1>();

    if((c1 == '[' || c1 == '{') && (c2 == ']' || c2 == '}')) {
      return l1 == l2;
    }
    return true;
  }
};

struct opening_and_closing_chars_are_corresponding
{
  __host__ __device__
  bool operator()(const adjacent_chars& x)
  {
    char c1 = x.get<0>();
    char c2 = x.get<1>();
    
    if(c1 == '[') {
      return c2 != '}';
    }
    if(c1 == '{') {
      return c2 != ']';
    }
    return true;
  }
};

void elapsedTime(chrono::time_point<chrono::steady_clock> start, string description);

#endif
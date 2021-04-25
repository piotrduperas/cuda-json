#ifndef CUDA_JSON_COMMON_CUH
#define CUDA_JSON_COMMON_CUH

#include <thrust/tuple.h>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

typedef thrust::tuple<char, int> char_and_position;

struct json_char {
  char _char;
  char type; // 1 for opening or -1 for closing
  int position; // position in file
  short level;
};

struct braces_to_numbers
{
  __host__ __device__
  json_char operator()(const char_and_position& x) const { 
    json_char c;
    c.position = x.get<1>();
    c._char = x.get<0>();

    switch(c._char) {
      case '{':
      case '[':
        c.type = 1;
        break;
      case '}':
      case ']':
        c.type = -1;
        break;
    }

    return c;
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

struct get_type_from_json_char
{
  __host__ __device__
  char operator()(const json_char& x) const { 
    return x.type;
  }
};

struct increment
{
  __host__ __device__
  json_char operator()(json_char& x) const { 
    x.level++;
    return x;
  }
};

struct is_closing_brace
{
  __host__ __device__
  bool operator()(const json_char& x) const { 
    return x.type == -1;
  }
};

void elapsedTime(chrono::time_point<chrono::steady_clock> start, string description);

#endif
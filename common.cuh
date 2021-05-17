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

typedef thrust::tuple<json_char, json_char> adjacent_chars;


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

struct get_type_from_json_char
{
  __host__ __device__
  char operator()(const json_char& x) const { 
    return x.type;
  }
};

struct get_value_from_char
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

struct assign_level_to_json_char
{
  __host__ __device__
  json_char operator()(json_char& x, const short& level) const { 
    x.level = level + x.type == 1 ? 0 : 1;
    return x;
  }
};

struct is_brace
{
  __host__ __device__
  bool operator()(const json_char& x)
  {
    return 
      x._char == '{' || 
      x._char == '}';
  }
};

struct is_bracket
{
  __host__ __device__
  bool operator()(const json_char& x)
  {
    return 
      x._char == '[' || 
      x._char == ']';
  }
};

struct opening_and_closing_chars_have_the_same_level
{
  __host__ __device__
  bool operator()(const adjacent_chars& x)
  {
    json_char c1 = x.get<0>();
    json_char c2 = x.get<1>();

    if(c1.type == 1 && c2.type == -1) {
      return c1.level == c2.level;
    }
    return true;
  }
};

struct opening_and_closing_chars_are_corresponding
{
  __host__ __device__
  bool operator()(const adjacent_chars& x)
  {
    json_char c1 = x.get<0>();
    json_char c2 = x.get<1>();
    
    if(c1._char == '[') {
      return c2._char != '}';
    }
    if(c1._char == '{') {
      return c2._char != ']';
    }
    return true;
  }
};

struct opening_and_closing_chars_have_the_same_level_and_are_corresponding
{
  __host__ __device__
  bool operator()(const adjacent_chars& x)
  {
    json_char c1 = x.get<0>();
    json_char c2 = x.get<1>();

    if(c1._char == '[') {
      if(c2._char != '}') return false;
    }
    if(c1._char == '{') {
      if(c2._char != '[') return false;
    }
    if(c1.type == 1 && c2.type == -1) {
      return c1.level == c2.level;
    }
    return true;
  }
};

void elapsedTime(chrono::time_point<chrono::steady_clock> start, string description);

#endif
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/partition.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <chrono>
#include "common.cuh"
#include <stack>

using namespace std;

bool is_opening_bracket(const char& c)
{
  if(c=='{' || c=='[') return true;
  return false;
}

bool is_closing_bracket(const char& c)
{
  if(c=='}' || c==']') return true;
  return false;
}

bool is_bracket_coresponding(const char& c, const char& c_stack)
{
  if(c=='}') 
  {
    if(c_stack=='{') return true;
  }
  else if(c==']')
  {
    if(c_stack=='[') return true;
  }
  return false;
}

__host__
bool h_is_balanced_parentheses(const string& s)
{
    stack<char> bracket_stack;
    int depth = 0;

    for (auto it = s.cbegin() ; it != s.cend(); ++it) 
    {
      if(is_opening_bracket(*it))
      {
        bracket_stack.push(*it);
        depth++;
      }
      else if(is_closing_bracket(*it))
      {
        if(is_bracket_coresponding(*it,bracket_stack.top()))
        {
          depth--;
          if(depth<0) return false;
          bracket_stack.pop();
        }
        else return false;
      }
    }
    if(depth==0) return true;
    return false;
}

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

int main(int argc, char **argv)
{
  cout << "CUDA JSON Validator" << endl << endl;
  cout << "Warming up" << endl << endl;
  warm_up_gpu<<<1024, 1024>>>();


  string result = "";

  auto startReadingFile = chrono::steady_clock::now();

  // Reading file
  const string& json_file = argc == 2 ? argv[1] : "samples/senators_short.json";
  ifstream ifile(json_file);

  ostringstream ss;
  ss << ifile.rdbuf();
  const string& s = ss.str();

  elapsedTime(startReadingFile, "Reading file");

  auto startCpu = chrono::steady_clock::now();
  //cpu
  string result_cpu = h_is_balanced_parentheses(s) ? "correct" : "wrong";
  cout << "\nCpu algorithm claims that file is " << result_cpu <<endl;
  elapsedTime(startCpu, "Cpu alogithm");

  auto startCopying = chrono::steady_clock::now();

  thrust::host_vector<char> H_file(s.begin(), s.end());
  thrust::device_vector<char> D_file(H_file.size());

  size_t file_size = H_file.size();

  // Copy file to device memory
  thrust::copy(H_file.begin(), H_file.end(), D_file.begin());

  elapsedTime(startCopying, "Copying file");
  auto startCalculations = chrono::steady_clock::now();

  // Vector for (brace, it's position) pair
  thrust::device_vector<char_and_position> D_braces_pos(file_size);

  // Zip file characters with their positions
  auto char_and_pos = thrust::make_zip_iterator(thrust::make_tuple(D_file.begin(), thrust::make_counting_iterator(0)));
  
  // Filter only braces
  auto last_brace_it = thrust::copy_if(char_and_pos, char_and_pos + s.length(), D_braces_pos.begin(), is_brace_or_bracket());
  D_braces_pos.resize(last_brace_it - D_braces_pos.begin());

  thrust::device_vector<json_char> D_json_chars(D_braces_pos.size());
  thrust::device_vector<short> D_levels(D_braces_pos.size());

  // Transform { to 1, } to -1
  thrust::transform(D_braces_pos.begin(), D_braces_pos.end(), D_json_chars.begin(), braces_to_numbers());

  // Calculate nesting levels of braces
  thrust::transform_inclusive_scan(D_json_chars.begin(), D_json_chars.end(), D_levels.begin(), get_type_from_json_char(), thrust::plus<short>());
  thrust::transform(D_json_chars.begin(), D_json_chars.end(), D_levels.begin(), D_json_chars.begin(), assign_level_to_json_char());
  // Add 1 to closing braces
  //thrust::transform_if(D_json_chars.begin(), D_json_chars.end(), D_json_chars.begin(), D_json_chars.begin(), increment(), is_closing_brace());

  char last_brace_level = ((json_char)D_json_chars[D_json_chars.size() - 1]).level;

  if(result != "" && last_brace_level != 1){
    stringstream tmp;
    tmp << "Braces or brackets in this JSON are incorrect. Last brace has level " << (int)last_brace_level << ", but should have level 1";
    result = tmp.str();
  }

  // Check if everything between corresponding { and } is correct
  //auto last_brace = thrust::copy_if(D_json_chars.begin(), D_json_chars.end(), D_one_type_only.begin(), is_brace());
  //int braces_count = last_brace - D_one_type_only.begin();

  //if(braces_count % 2){
  //  result = "Uneven number of braces";
  //} else if(braces_count > 0){
    //auto adjacent_braces = thrust::make_zip_iterator(thrust::make_tuple(D_one_type_only.begin(), D_one_type_only.begin() + 1));
    //bool are_braces_correct = thrust::all_of(adjacent_braces, adjacent_braces + braces_count - 1, opening_and_closing_chars_have_the_same_level());

    //if(!are_braces_correct){
    //  result = "Something between some braces is incorrect";
    //}
  //}

  // Check if everything between corresponding [ and ] is correct
  //auto last_bracket = thrust::copy_if(D_json_chars.begin(), D_json_chars.end(), last_brace, is_bracket());
  //int brackets_count = last_bracket - D_one_type_only.begin();

  //if(brackets_count % 2){
  //  result = "Uneven number of brackets";
  //} else if(brackets_count > 0){
    thrust::partition(D_json_chars.begin(), D_json_chars.end(), is_closing_brace());
    auto adjacent_brackets = thrust::make_zip_iterator(thrust::make_tuple(D_json_chars.begin(), D_json_chars.begin() + 1));
    bool are_brackets_correct = thrust::all_of(adjacent_brackets, adjacent_brackets + D_json_chars.size(), opening_and_closing_chars_have_the_same_level());

    if(!are_brackets_correct){
      result = "Something between some brackets is incorrect";
    }
  //}

  auto adjacent_chars = thrust::make_zip_iterator(thrust::make_tuple(D_json_chars.begin(), D_json_chars.begin() + 1));
  bool are_chars_correct = thrust::all_of(adjacent_chars, adjacent_chars + D_json_chars.size() - 1, opening_and_closing_chars_are_corresponding());

  if(!are_chars_correct){
    result = "Found sequence [} or {]";
  }

  elapsedTime(startCalculations, "Calculations");

  /*
  for(int i = 0; i < D_json_chars.size(); i++){
    json_char c = (json_char)D_json_chars[i];
    cout << c.position << " - " << c._char << " - " << c.level << endl;
  }
  */


 

  if(result != ""){
    cout << "JSON is incorrect:\n\t" << result << endl;
  } else {
    cout << "JSON is correct" << endl;
  }
  
  return 0;
}

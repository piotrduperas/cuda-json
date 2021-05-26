#include <chrono>
#include <iostream>
#include "simdjson.h"
#include "time_measurements.h"
using namespace simdjson;
int main(int argc, char **argv){
    ondemand::parser parser;
    auto calculations = std::chrono::steady_clock::now();

    padded_string json = padded_string::load(argv[1]);
    
    ondemand::document tweets = parser.iterate(json);

    try {
      std::cout << (double)tweets.find_field("sdf") << " results." << std::endl;
    } catch(...){

    }

    elapsedTime(calculations, "Calculations");
}
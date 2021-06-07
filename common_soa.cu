#include "common_soa.cuh"

void elapsedTime(chrono::time_point<chrono::steady_clock> start, string description) {
  auto end = std::chrono::steady_clock::now();

  cout << description << ": "
    << chrono::duration_cast<chrono::milliseconds>(end - start).count()
    << "ms" << endl;
}
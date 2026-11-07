// SPDX-License-Identifier: MIT
#include "radial_executor.h"

#include <stdexcept>
#include <vector>

int main() {
  for (int threads : {1, 4, 8}) {
    RadialExecutor executor(threads);
    std::vector<int> visits(99, 0);
    for (int repeat = 0; repeat < 1000; ++repeat) {
      executor.Run(0, 99, [&](int index) { ++visits[index]; });
    }
    for (int value : visits) {
      if (value != 1000) {
        return 1;
      }
    }
    for (int failing_index : {0, 1, 17}) {
      bool caught = false;
      try {
        executor.Run(0, 99, [failing_index](int index) {
          if (index == failing_index) {
            throw std::runtime_error("test");
          }
        });
      } catch (const std::runtime_error&) {
        caught = true;
      }
      if (!caught) {
        return 2;
      }
      executor.Run(0, 99, [&](int index) { visits[index] = 7; });
      for (int value : visits) {
        if (value != 7) {
          return 3;
        }
      }
    }
    executor.Run(0, 0, [](int) { throw std::runtime_error("empty"); });
  }
  return 0;
}

// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_RADIAL_EXECUTOR_H_
#define VMECPP_EXPERIMENT_RADIAL_EXECUTOR_H_

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// Each call owns disjoint radial outputs. Reduction order within an output
// stays unchanged. Only the submitting thread may call Run.
class RadialExecutor {
 public:
  explicit RadialExecutor(int count) {
    try {
      for (int index = 1; index < count; ++index) {
        threads_.emplace_back([this, index] { Worker(index); });
      }
    } catch (...) {
      Stop();
      throw;
    }
  }
  ~RadialExecutor() { Stop(); }
  RadialExecutor(const RadialExecutor&) = delete;
  RadialExecutor& operator=(const RadialExecutor&) = delete;

  void Run(int first, int last, const std::function<void(int)>& task) {
    if (threads_.empty()) {
      for (int index = first; index < last; ++index) {
        task(index);
      }
      return;
    }
    {
      std::lock_guard lock(mutex_);
      first_ = first;
      last_ = last;
      task_ = task;
      error_ = nullptr;
      remaining_ = threads_.size();
      ++generation_;
    }
    ready_.notify_all();
    try {
      Execute(0);
    } catch (...) {
      RecordError();
    }
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return remaining_ == 0; });
    task_ = nullptr;
    const auto error = error_;
    lock.unlock();
    if (error) {
      std::rethrow_exception(error);
    }
  }

 private:
  void Execute(int worker) {
    const int stride = static_cast<int>(threads_.size()) + 1;
    for (int index = first_ + worker; index < last_; index += stride) {
      task_(index);
    }
  }
  void RecordError() {
    std::lock_guard lock(mutex_);
    if (!error_) {
      error_ = std::current_exception();
    }
  }
  void Worker(int index) {
    std::size_t observed = 0;
    for (;;) {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this, observed] {
        return stopping_ || generation_ != observed;
      });
      if (stopping_) {
        return;
      }
      observed = generation_;
      lock.unlock();
      try {
        Execute(index);
      } catch (...) {
        RecordError();
      }
      lock.lock();
      if (--remaining_ == 0) {
        done_.notify_one();
      }
    }
  }
  void Stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable done_;
  std::function<void(int)> task_;
  std::exception_ptr error_;
  std::size_t generation_ = 0;
  std::size_t remaining_ = 0;
  int first_ = 0;
  int last_ = 0;
  bool stopping_ = false;
};

#endif  // VMECPP_EXPERIMENT_RADIAL_EXECUTOR_H_

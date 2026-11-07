// SPDX-License-Identifier: MIT
#ifndef VMECPP_EXPERIMENT_PINNED_CPU_LOG_GUARD_H_
#define VMECPP_EXPERIMENT_PINNED_CPU_LOG_GUARD_H_

#include <dlfcn.h>
#include <immintrin.h>

#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

struct PinnedLogFpEnvironment {
  int rounding_;
  unsigned int mxcsr_;
  unsigned short x87_control_;
};
inline PinnedLogFpEnvironment ReadPinnedLogFpEnvironment() {
  PinnedLogFpEnvironment result{std::fegetround(), _mm_getcsr(), 0};
  asm volatile("fnstcw %0" : "=m"(result.x87_control_));
  return result;
}
inline bool PinnedLogFpEnvironmentSupported(
    const PinnedLogFpEnvironment& environment) {
  // Preserve existing modes; status flags in MXCSR's low six bits are sticky.
  return environment.rounding_ == FE_TONEAREST &&
         (environment.mxcsr_ & 0xffffffc0U) == 0x1f80U &&
         environment.x87_control_ == 0x037f;
}
enum class PinnedLogGuardReason {
  SUPPORTED,
  UNKNOWN_EXTENSION,
  UNKNOWN_PROVIDER,
  UNKNOWN_WRAPPER,
  UNKNOWN_INNER,
  UNKNOWN_CODE_OR_TABLE,
  UNREADABLE,
  HASH_UNAVAILABLE,
  FP_ENVIRONMENT,
  BINDING_CHANGED,
  COUNT
};
inline const char* PinnedLogGuardReasonName(PinnedLogGuardReason reason) {
  constexpr std::array<const char*, 10> kNames{
      "supported",       "unknown_extension", "unknown_provider",
      "unknown_wrapper", "unknown_inner",     "unknown_code_or_table",
      "unreadable",      "hash_unavailable",  "fp_environment",
      "binding_changed"};
  return kNames.at(static_cast<std::size_t>(reason));
}
struct PinnedLogGuardDecision {
  bool allowed_;
  PinnedLogGuardReason reason_;
  PinnedLogFpEnvironment environment_;
};

class PinnedCpuLogGuard {
 public:
  explicit PinnedCpuLogGuard(void* original_evolve) {
    const int saved_errno = errno;
    std::fenv_t saved_environment;
    const bool saved = std::fegetenv(&saved_environment) == 0;
    try {
      Initialize(original_evolve);
    } catch (...) {
      identity_reason_ = PinnedLogGuardReason::UNREADABLE;
    }
    if (!saved || std::fesetenv(&saved_environment) != 0)
      identity_reason_ = PinnedLogGuardReason::FP_ENVIRONMENT;
    errno = saved_errno;
  }
  PinnedLogGuardDecision Evaluate() const {
    const auto environment = ReadPinnedLogFpEnvironment();
    if (identity_reason_ != PinnedLogGuardReason::SUPPORTED)
      return {false, identity_reason_, environment};
    if (!PinnedLogFpEnvironmentSupported(environment))
      return {false, PinnedLogGuardReason::FP_ENVIRONMENT, environment};
    // Both owning modules remain loaded through their existing oracle handles.
    // Recheck the two actual dispatch slots on every accepted controller call.
    if (ReadPointer(extension_slot_) != wrapper_ ||
        ReadPointer(inner_slot_) != inner_)
      return {false, PinnedLogGuardReason::BINDING_CHANGED, environment};
    return {true, PinnedLogGuardReason::SUPPORTED, environment};
  }
  const std::string& ProviderSha256() const { return provider_sha256_; }
  const std::string& ProviderPath() const { return provider_path_; }
  const std::string& HashProviderPath() const { return hash_provider_path_; }
  const std::string& HashProviderSha256() const {
    return hash_provider_sha256_;
  }
  PinnedLogGuardReason IdentityReason() const { return identity_reason_; }
  // Use the actual extension relocation for CPU escape, not this shim's log.
  using CpuLogFunction = double (*)(double);
  CpuLogFunction ActualCpuLog() const {
    if (identity_reason_ != PinnedLogGuardReason::SUPPORTED) return nullptr;
    return reinterpret_cast<CpuLogFunction>(ReadPointer(extension_slot_));
  }

 private:
  using Sha256 = unsigned char* (*)(const unsigned char*, std::size_t,
                                    unsigned char*);
  struct Stack {
    const char* file_sha256_;
    std::uintptr_t wrapper_;
    std::uintptr_t got_;
    std::uintptr_t inner_;
    std::uintptr_t table_;
    const char* wrapper_sha256_;
    const char* inner_sha256_;
  };
  static bool Readable(std::uintptr_t address, std::size_t bytes,
                       bool executable = false) {
    if (!address || address + bytes < address) return false;
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
      unsigned long start = 0, end = 0;
      std::array<char, 5> permissions{};
      if (std::sscanf(line.c_str(), "%lx-%lx %4s", &start, &end,
                      permissions.data()) == 3 &&
          start <= address && address + bytes <= end && permissions[0] == 'r' &&
          (!executable || permissions[2] == 'x'))
        return true;
    }
    return false;
  }
  static std::uintptr_t ReadPointer(std::uintptr_t address) {
    return *reinterpret_cast<const volatile std::uintptr_t*>(address);
  }
  std::string Hash(const void* data, std::size_t size) const {
    std::array<unsigned char, 32> digest{};
    if (sha256_(static_cast<const unsigned char*>(data), size, digest.data()) ==
        nullptr)
      return {};
    std::string result;
    constexpr std::string_view kDigits = "0123456789abcdef";
    for (const auto byte : digest) {
      result += kDigits[byte >> 4];
      result += kDigits[byte & 15];
    }
    return result;
  }
  std::string HashFile(const char* path) const {
    if (path == nullptr) return {};
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() > 128 * 1024 * 1024)
      return {};
    file.seekg(0);
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(file),
                                           std::istreambuf_iterator<char>()};
    return Hash(bytes.data(), bytes.size());
  }
  bool MatchMemory(std::uintptr_t address, std::size_t bytes,
                   const char* expected, bool executable) const {
    return Readable(address, bytes, executable) &&
           Hash(reinterpret_cast<const void*>(address), bytes) == expected;
  }
  void Initialize(void* original_evolve) {
    // Optional hash service is resolved only for this opt-in guard. Absence
    // declines the exact-log path instead of adding a mandatory solver DSO.
    static void* crypto = dlopen("libcrypto.so.3", RTLD_NOW | RTLD_LOCAL);
    if (crypto != nullptr)
      sha256_ = reinterpret_cast<Sha256>(dlsym(crypto, "SHA256"));
    if (sha256_ == nullptr) {
      identity_reason_ = PinnedLogGuardReason::HASH_UNAVAILABLE;
      return;
    }
    Dl_info hash_info{};
    if (dladdr(reinterpret_cast<void*>(sha256_), &hash_info))
      hash_provider_path_ = hash_info.dli_fname;
    hash_provider_sha256_ = HashFile(hash_provider_path_.c_str());
    Dl_info reference{};
    if (!dladdr(original_evolve, &reference) ||
        HashFile(reference.dli_fname) !=
            "144e4f204cbb0569edd25710cf8ef4c0e618b0e7a60e33cf3a556066e8a06da"
            "8") {
      identity_reason_ = PinnedLogGuardReason::UNKNOWN_EXTENSION;
      return;
    }
    extension_slot_ =
        reinterpret_cast<std::uintptr_t>(reference.dli_fbase) + 0x2dc4f8;
    if (!Readable(extension_slot_, sizeof(std::uintptr_t))) return;
    wrapper_ = ReadPointer(extension_slot_);
    Dl_info provider{};
    if (!dladdr(reinterpret_cast<void*>(wrapper_), &provider)) return;
    provider_path_ = provider.dli_fname;
    provider_sha256_ = HashFile(provider.dli_fname);
    constexpr std::array<Stack, 2> kStacks{
        {{"fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0",
          0x3a5f0, 0xe7f40, 0x79d50, 0xb5240,
          "464df4564afe4966f1591ebc7973b0e94a054b00376e07e7b8ae5c0e330af853",
          "17e98c6f826879db72e0d7bfa48188e752c6e6a97857c58f01faa65cedb97685"},
         {"1c488db6f4fe8cecb72456bc018de000b7a9b539c2994864b0416b5e8371b3d8",
          0x5b100, 0x124f20, 0xa11c0, 0xe71e0,
          "e122c9aa45ae0e97c3c36754cd3ea9d3d70e9855701ece6911ebff6227e683a5",
          "d5baaf8ef01cfe6f0ebf397490091168ee32a2b9f091679c91fdb4e6c6f9c858"}}};
    for (const auto& stack : kStacks) {
      if (provider_sha256_ != stack.file_sha256_) continue;
      const auto base = reinterpret_cast<std::uintptr_t>(provider.dli_fbase);
      if (wrapper_ != base + stack.wrapper_) {
        identity_reason_ = PinnedLogGuardReason::UNKNOWN_WRAPPER;
        return;
      }
      inner_slot_ = base + stack.got_;
      if (!Readable(inner_slot_, sizeof(std::uintptr_t))) return;
      inner_ = ReadPointer(inner_slot_);
      if (inner_ != base + stack.inner_) {
        identity_reason_ = PinnedLogGuardReason::UNKNOWN_INNER;
        return;
      }
      if (!MatchMemory(wrapper_, 57, stack.wrapper_sha256_, true) ||
          !MatchMemory(inner_, 570, stack.inner_sha256_, true) ||
          !MatchMemory(base + stack.table_, 2192,
                       "d8f77f04619b10fe29031a98a0596af699fd6329c5253affc554092"
                       "a17cf1bd8",
                       false)) {
        identity_reason_ = PinnedLogGuardReason::UNKNOWN_CODE_OR_TABLE;
        return;
      }
      identity_reason_ = PinnedLogGuardReason::SUPPORTED;
      return;
    }
    identity_reason_ = PinnedLogGuardReason::UNKNOWN_PROVIDER;
  }
  Sha256 sha256_ = nullptr;
  PinnedLogGuardReason identity_reason_ = PinnedLogGuardReason::UNREADABLE;
  std::uintptr_t extension_slot_ = 0, wrapper_ = 0, inner_slot_ = 0, inner_ = 0;
  std::string provider_sha256_;
  std::string provider_path_;
  std::string hash_provider_path_;
  std::string hash_provider_sha256_;
};
#endif  // VMECPP_EXPERIMENT_PINNED_CPU_LOG_GUARD_H_

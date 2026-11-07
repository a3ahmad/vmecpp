// Diagnostic-only observer of the actual CPU libm log input/output bits.
// This forwards to RTLD_NEXT without replacing any numerical operation.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace {

struct LogRecord {
  std::uint64_t magic_ = 0x564d45434c4f4731ULL;
  std::uint64_t sequence_ = 0;
  std::uint64_t thread_id_ = 0;
  std::uint64_t caller_offset_ = 0;
  std::uint64_t input_bits_ = 0;
  std::uint64_t output_bits_ = 0;
  std::uint64_t errno_ = 0;
  std::uint64_t exceptions_ = 0;
  std::array<char, 256> module_{};
  std::array<char, 128> symbol_{};
};
static_assert(sizeof(LogRecord) == 448);

[[noreturn]] void Fail() {
  constexpr std::string_view kMessage =
      "CPU log observer failed; diagnostic aborted\n";
  const auto ignored = write(STDERR_FILENO, kMessage.data(), kMessage.size());
  static_cast<void>(ignored);
  _exit(122);
}

int OpenOutput() {
  const char* prefix = std::getenv("VMECPP_CPU_LOG_OBSERVER_PREFIX");
  if (prefix == nullptr || prefix[0] == '\0') return -1;
  std::array<char, 4096> path{};
  const int length = std::snprintf(path.data(), path.size(), "%s.%ld.bin",
                                   prefix, static_cast<long>(getpid()));
  if (length < 0 || static_cast<std::size_t>(length) >= path.size()) Fail();
  const int output = open(path.data(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
  if (output < 0) Fail();
  return output;
}

template <std::size_t kSize>
void CopyName(const char* name, std::array<char, kSize>& m_output) {
  if (name == nullptr) return;
  const std::size_t length = std::strlen(name);
  // Never silently truncate a module or symbol needed to identify the caller.
  if (length >= kSize) Fail();
  std::memcpy(m_output.data(), name, length);
}

}  // namespace

extern "C" double log(double value) noexcept {
  using LogFunction = double (*)(double);
  const int original_errno = errno;
  static const auto original =
      reinterpret_cast<LogFunction>(dlsym(RTLD_NEXT, "log"));
  if (original == nullptr) Fail();
  errno = original_errno;
  const double result = original(value);
  const int result_errno = errno;
  std::fenv_t environment;
  if (std::fegetenv(&environment) != 0) Fail();
  static thread_local bool recording = false;
  if (!recording) {
    recording = true;
    static const int output = OpenOutput();
    if (output >= 0) {
      static thread_local std::uint64_t sequence = 0;
      LogRecord record;
      record.sequence_ = sequence++;
      record.thread_id_ = static_cast<std::uint64_t>(syscall(SYS_gettid));
      std::memcpy(&record.input_bits_, &value, sizeof(value));
      std::memcpy(&record.output_bits_, &result, sizeof(result));
      record.errno_ = static_cast<std::uint64_t>(result_errno);
      record.exceptions_ =
          static_cast<std::uint64_t>(std::fetestexcept(FE_ALL_EXCEPT));
      void* caller = __builtin_return_address(0);
      Dl_info info{};
      if (dladdr(caller, &info) == 0) Fail();
      record.caller_offset_ = reinterpret_cast<std::uintptr_t>(caller) -
                              reinterpret_cast<std::uintptr_t>(info.dli_fbase);
      CopyName(info.dli_fname, record.module_);
      CopyName(info.dli_sname, record.symbol_);
      // One write per record on a shared regular-file descriptor. Short writes
      // abort instead of producing an apparently valid partial observation.
      ssize_t written;
      do {
        written = write(output, &record, sizeof(record));
      } while (written < 0 && errno == EINTR);
      if (written != sizeof(record)) Fail();
    }
    recording = false;
  }
  if (std::fesetenv(&environment) != 0) Fail();
  errno = result_errno;
  return result;
}

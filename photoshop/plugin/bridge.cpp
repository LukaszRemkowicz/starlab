#include "bridge.h"

#include <atomic>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <CoreFoundation/CoreFoundation.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace selb {

static std::atomic<unsigned> gCounter{0};

static void put32(uint8_t *p, uint32_t v) {  // header is little-endian; all supported targets are LE
    std::memcpy(p, &v, 4);
}

static std::string uniqueName() {
#ifdef _WIN32
    unsigned pid = GetCurrentProcessId();
#else
    unsigned pid = unsigned(getpid());
#endif
    return "starlab-" + std::to_string(pid) + "-" + std::to_string(++gCounter);  // < 30 chars
}

static void writeHeader(Block &b, uint32_t maxValue) {
    std::memcpy(b.base, "SELB", 4);
    put32(b.base + 4, kProtocolVersion);
    put32(b.base + 8, b.width);
    put32(b.base + 12, b.height);
    put32(b.base + 16, b.channels);
    put32(b.base + 20, b.depth);
    put32(b.base + 24, maxValue);
    put32(b.base + 28, kStatusReady);
    put32(b.base + kModeOffset, kModeInteractive);
    put32(b.base + kSettingsLengthOffset, 0);
}

uint32_t settingsLength(const Block &b) {
    uint32_t v;
    std::memcpy(&v, b.base + kSettingsLengthOffset, 4);
    return v;
}

bool setRequest(Block &b, uint32_t mode, const uint8_t *json, uint32_t n) {
    if (n > kSettingsCapacity) return false;
    if (n) std::memcpy(b.settings(), json, n);
    put32(b.base + kModeOffset, mode);
    put32(b.base + kSettingsLengthOffset, n);
    return true;
}

uint32_t status(const Block &b) {
    uint32_t v;
    std::memcpy(&v, b.base + 28, 4);
    return v;
}

#ifdef _WIN32

bool create(Block &b, uint32_t width, uint32_t height, uint32_t channels, uint32_t depth,
            uint32_t maxValue, std::string &err) {
    b.width = width; b.height = height; b.channels = channels; b.depth = depth;
    b.size = kHeaderSize + b.rowBytes() * height + kSettingsCapacity;
    b.name = uniqueName();
    std::wstring wname(b.name.begin(), b.name.end());  // ASCII only
    unsigned long long size = b.size;
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, DWORD(size >> 32),
                                  DWORD(size & 0xFFFFFFFFu), wname.c_str());
    if (!h) { err = "Could not allocate shared memory for the image."; return false; }
    b.handle = h;
    b.base = static_cast<uint8_t *>(MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 0));
    if (!b.base) { err = "Could not map shared memory for the image."; destroy(b); return false; }
    writeHeader(b, maxValue);
    return true;
}

void destroy(Block &b) {
    if (b.base) UnmapViewOfFile(b.base);
    if (b.handle) CloseHandle(static_cast<HANDLE>(b.handle));
    b.base = nullptr; b.handle = nullptr;
}

static std::wstring widen(const std::string &s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

// shortcut: quoting only escapes '"'; fine for paths and our fixed arguments, upgrade if args get backslash-quote mixes.
static std::wstring quote(const std::string &s) {
    std::wstring out = L"\"";
    for (wchar_t c : widen(s)) { if (c == L'"') out += L'\\'; out += c; }
    return out + L"\"";
}

bool run(const std::string &exe, const std::vector<std::string> &args, int &exitCode, std::string &err) {
    std::wstring cmd = quote(exe);
    for (const auto &a : args) cmd += L" " + quote(a);
    STARTUPINFOW si{}; si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        err = "Could not start Star Effects Lab (" + exe + "). Is it installed?";
        return false;
    }
    for (;;) {  // wait for exit but keep pumping Photoshop's messages
        DWORD r = MsgWaitForMultipleObjects(1, &pi.hProcess, FALSE, INFINITE, QS_ALLINPUT);
        if (r == WAIT_OBJECT_0) break;
        if (r == WAIT_FAILED) { err = "Waiting for Star Effects Lab failed."; break; }
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    }
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    if (!err.empty()) return false;
    exitCode = int(code);
    return true;
}

#else  // macOS

bool create(Block &b, uint32_t width, uint32_t height, uint32_t channels, uint32_t depth,
            uint32_t maxValue, std::string &err) {
    b.width = width; b.height = height; b.channels = channels; b.depth = depth;
    b.size = kHeaderSize + b.rowBytes() * height + kSettingsCapacity;
    b.name = uniqueName();
    b.fd = shm_open(("/" + b.name).c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (b.fd < 0) { err = "Could not allocate shared memory for the image."; return false; }
    if (ftruncate(b.fd, off_t(b.size)) != 0) { err = "Not enough memory to share the image."; destroy(b); return false; }
    void *p = mmap(nullptr, b.size, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
    if (p == MAP_FAILED) { err = "Could not map shared memory for the image."; destroy(b); return false; }
    b.base = static_cast<uint8_t *>(p);
    close(b.fd); b.fd = -1;  // mapping stays valid
    writeHeader(b, maxValue);
    return true;
}

void destroy(Block &b) {
    if (b.base) munmap(b.base, b.size);
    if (b.fd >= 0) close(b.fd);
    if (!b.name.empty()) shm_unlink(("/" + b.name).c_str());
    b.base = nullptr; b.fd = -1; b.name.clear();
}

bool run(const std::string &exe, const std::vector<std::string> &args, int &exitCode, std::string &err) {
    std::vector<char *> argv{const_cast<char *>(exe.c_str())};
    for (const auto &a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid;
    int rc = posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        err = "Could not start Star Effects Lab (" + exe + "). Is it installed?";
        return false;
    }
    int st = 0;
    for (;;) {  // poll so the host's run loop (and its UI) keeps running meanwhile
        pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) break;
        if (r < 0) { err = "Waiting for Star Effects Lab failed."; return false; }
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, true);
    }
    exitCode = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return true;
}

#endif

}  // namespace selb

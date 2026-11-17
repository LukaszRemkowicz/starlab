// Shared-memory exchange with the Star Effects Lab app (protocol: src/starlab/infrastructure/photoshop_bridge.py).
// No Photoshop headers here, so the self-test can use it without the SDK.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace selb {

constexpr uint32_t kHeaderSize = 64;
constexpr uint32_t kProtocolVersion = 2;
constexpr uint32_t kModeOffset = 32, kSettingsLengthOffset = 36;
constexpr uint32_t kModeInteractive = 0, kModeRepeat = 1;
constexpr size_t kSettingsCapacity = 65536;  // JSON area right after the pixels
constexpr uint32_t kStatusReady = 0, kStatusWritten = 1, kStatusCancelled = 2;

struct Block {
    std::string name;       // name Python is given (no leading "/")
    uint8_t *base = nullptr;
    size_t size = 0;
    void *handle = nullptr; // Windows: mapping HANDLE; unused on macOS
    int fd = -1;            // macOS only
    uint32_t width = 0, height = 0, channels = 0, depth = 0;

    uint8_t *pixels() const { return base + kHeaderSize; }
    uint8_t *settings() const { return pixels() + rowBytes() * height; }
    size_t rowBytes() const { return size_t(width) * channels * (depth / 8); }
};

// Create a uniquely named block with a valid header (status 0). False + err on failure.
bool create(Block &b, uint32_t width, uint32_t height, uint32_t channels, uint32_t depth,
            uint32_t maxValue, std::string &err);

// Run `exe args...` and wait for it, keeping the host UI responsive. exitCode is -1 for a signal/crash.
bool run(const std::string &exe, const std::vector<std::string> &args, int &exitCode, std::string &err);

uint32_t status(const Block &b);
uint32_t settingsLength(const Block &b);

// Set the mode word (0 interactive, 1 repeat) and the settings JSON (n <= kSettingsCapacity) before running the app.
bool setRequest(Block &b, uint32_t mode, const uint8_t *json, uint32_t n);

// Unmap and unlink/close the block. Safe on a half-created block.
void destroy(Block &b);

}  // namespace selb

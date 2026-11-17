// Star Effects Lab Photoshop filter: hands the layer to the app through shared memory (no disk) and
// copies the result back. Own implementation against the Adobe SDK headers; no SDK sample code.
#include "PIDefines.h"
#include "PITypes.h"
#include "PIAbout.h"
#include "PIFilter.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bridge.h"

// Absolute path of the launcher; set by CMake (-DSTARLAB_COMMAND=...). The installer will set the real one.
#ifndef STARLAB_COMMAND
#define STARLAB_COMMAND "star-effects-lab"
#endif

namespace {

constexpr int64 kMaxHostSpace = 64 * 1024 * 1024;
constexpr size_t kBandBytes = 16 * 1024 * 1024;

FilterRecord *gRec = nullptr;
bool gBig = false;  // using the 32-bit-coordinate fields
bool gShowDialog = false;  // the Parameters selector ran in this invocation (not "Last Filter")

// Last settings JSON, kept by Photoshop in gRec->parameters between invocations: u32 length + bytes.
std::string storedSettings() {
    Handle h = gRec->parameters;
    if (!h || !gRec->handleProcs) return {};
    std::string out;
    if (gRec->handleProcs->getSizeProc(h) >= int32(sizeof(uint32_t))) {
        const char *p = gRec->handleProcs->lockProc(h, false);
        if (!p) return {};
        uint32_t n;
        std::memcpy(&n, p, sizeof n);
        if (n <= selb::kSettingsCapacity && size_t(n) + sizeof n <= size_t(gRec->handleProcs->getSizeProc(h)))
            out.assign(p + sizeof n, n);
        gRec->handleProcs->unlockProc(h);
    }
    return out;
}

void storeSettings(const uint8_t *json, uint32_t n) {
    if (!gRec->handleProcs) return;
    Handle h = gRec->handleProcs->newProc(int32(sizeof n + n));
    if (!h) return;  // keep the old settings
    char *p = gRec->handleProcs->lockProc(h, false);
    if (!p) {  // keep the old settings
        gRec->handleProcs->disposeProc(h);
        return;
    }
    std::memcpy(p, &n, sizeof n);
    std::memcpy(p + sizeof n, json, n);
    gRec->handleProcs->unlockProc(h);
    if (gRec->parameters) gRec->handleProcs->disposeProc(gRec->parameters);
    gRec->parameters = h;
}

// Set fields by name: the SDK's C++ VRect constructor takes (left, top, right, bottom).
VRect rect(int32 top, int32 left, int32 bottom, int32 right) {
    VRect r;
    r.top = top; r.left = left; r.bottom = bottom; r.right = right;
    return r;
}

VRect filterRect() {
    if (gBig) return gRec->bigDocumentData->filterRect32;
    const Rect &r = gRec->filterRect;
    return rect(r.top, r.left, r.bottom, r.right);
}
VRect inRect() {
    if (gBig) return gRec->bigDocumentData->inRect32;
    const Rect &r = gRec->inRect;
    return rect(r.top, r.left, r.bottom, r.right);
}
VRect outRect() {
    if (gBig) return gRec->bigDocumentData->outRect32;
    const Rect &r = gRec->outRect;
    return rect(r.top, r.left, r.bottom, r.right);
}
void setRects(const VRect &in, const VRect &out) {
    if (gBig) {
        gRec->bigDocumentData->inRect32 = in;
        gRec->bigDocumentData->outRect32 = out;
        return;
    }
    gRec->inRect = Rect{int16(in.top), int16(in.left), int16(in.bottom), int16(in.right)};
    gRec->outRect = Rect{int16(out.top), int16(out.left), int16(out.bottom), int16(out.right)};
}

// Step log for diagnosing host interaction (sizes and error codes only, never pixels).
void trace(const char *fmt, ...) {
#ifdef _WIN32
    const char *base = std::getenv("LOCALAPPDATA");
    std::string path = std::string(base ? base : ".") + "\\Star Effects Lab Plugin.log";
#else
    const char *base = std::getenv("HOME");
    std::string path = std::string(base ? base : ".") + "/Library/Logs/Star Effects Lab Plugin.log";
#endif
    FILE *f = std::fopen(path.c_str(), "a");
    if (!f) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(f, fmt, args);
    va_end(args);
    std::fputc('\n', f);
    std::fclose(f);
}

// Host-facing error text: Photoshop shows "Cannot complete operation because " + message.
int16 fail(const std::string &msg) {
    trace("fail: %s", msg.c_str());
    if (gRec->errorString) {
        size_t n = std::min<size_t>(msg.size(), 255);
        (*gRec->errorString)[0] = uint8(n);
        std::memcpy(*gRec->errorString + 1, msg.data(), n);
        return errReportString;
    }
    return -1;
}

enum class Dir { In, Out };

// Move the filter rect between host and block in row bands; interleaved planes 0..channels-1.
int16 transfer(selb::Block &b, Dir dir) {
    const VRect fr = filterRect();
    const size_t rowBytes = b.rowBytes();
    const int32 bandRows = int32(std::max<size_t>(1, kBandBytes / rowBytes));
    const VRect none = rect(0, 0, 0, 0);
    gRec->inputRate = int32(1) << 16;
    for (int32 top = fr.top; top < fr.bottom;) {
        VRect band = rect(top, fr.left, std::min(fr.bottom, top + bandRows), fr.right);
        if (dir == Dir::In) {
            setRects(band, none);
            gRec->inLoPlane = 0; gRec->inHiPlane = int16(b.channels - 1);
        } else {
            setRects(none, band);
            gRec->outLoPlane = 0; gRec->outHiPlane = int16(b.channels - 1);
        }
        int16 err = gRec->advanceState();
        trace("advanceState %s rows %d-%d planes %d err %d inRowBytes %d outRowBytes %d", dir == Dir::In ? "in" : "out",
              band.top, band.bottom, int(b.channels), int(err), int(gRec->inRowBytes), int(gRec->outRowBytes));
        if (err != noErr) return err;
        if (gRec->abortProc && gRec->abortProc()) return userCanceledErr;
        if (dir == Dir::In) {
            // The host may hand back less than asked for; never trust more than it reports.
            VRect got = inRect();
            if (got.top != top || got.left != fr.left || got.right != fr.right || got.bottom <= top ||
                got.bottom > band.bottom)
                return fail("Photoshop returned an unexpected image area.");
            band.bottom = got.bottom;
        }
        if (dir == Dir::In) {
            if (!gRec->inData || gRec->inRowBytes < 0 || size_t(gRec->inRowBytes) < rowBytes)
                return fail("Photoshop returned unexpected image data.");
        } else {
            const VRect o = outRect();
            if (!gRec->outData || gRec->outRowBytes < 0 || size_t(gRec->outRowBytes) < rowBytes ||
                o.top != band.top || o.left != band.left || o.bottom != band.bottom || o.right != band.right)
                return fail("Photoshop returned unexpected image data.");
        }
        const uint8 *src = static_cast<const uint8 *>(gRec->inData);
        uint8 *dst = static_cast<uint8 *>(gRec->outData);
        for (int32 y = 0; y < band.bottom - band.top; ++y) {
            uint8 *blockRow = b.pixels() + size_t(band.top - fr.top + y) * rowBytes;
            if (dir == Dir::In)
                std::memcpy(blockRow, src + size_t(y) * gRec->inRowBytes, rowBytes);
            else
                std::memcpy(dst + size_t(y) * gRec->outRowBytes, blockRow, rowBytes);
        }
        // An output band is committed by the next advanceState (or when Start returns).
        top = band.bottom;
    }
    return noErr;
}

int16 start() {
    uint32 channels, depth, maxValue;
    switch (gRec->imageMode) {
        case plugInModeRGBColor: case plugInModeRGB48: channels = 3; break;
        case plugInModeGrayScale: case plugInModeGray16: channels = 1; break;
        default: return fail("only RGB and Grayscale images are supported.");
    }
    if (gRec->planes < int16(channels))
        return fail("the target has fewer channels than the image; select the composite RGB or Gray channel.");
    if (gRec->depth == 8) { depth = 8; maxValue = 255; }
    else if (gRec->depth == 16) { depth = 16; maxValue = 32768; }  // Photoshop's 16-bit white
    else return fail("only 8-bit and 16-bit images are supported.");

    const VRect fr = filterRect();
    const int32 w = fr.right - fr.left, h = fr.bottom - fr.top;
    trace("start: mode %d depth %d big %d rect %d,%d-%d,%d planes %d", int(gRec->imageMode), int(gRec->depth),
          int(gBig), fr.left, fr.top, fr.right, fr.bottom, int(gRec->planes));
    if (w < 2 || h < 2) return fail("the selection must be at least 2 x 2 pixels.");

    selb::Block block;
    std::string msg;
    if (!selb::create(block, uint32(w), uint32(h), channels, depth, maxValue, msg)) return fail(msg);
    struct Guard {  // frees the block on every exit, including exceptions (destroy is idempotent)
        selb::Block &b;
        ~Guard() { selb::destroy(b); }
    } guard{block};

    // Last Filter (no Parameters selector) with remembered settings repeats them without a window.
    const std::string last = storedSettings();
    const bool repeat = !gShowDialog && !last.empty();
    gShowDialog = false;
    if (!selb::setRequest(block, repeat ? selb::kModeRepeat : selb::kModeInteractive,
                          reinterpret_cast<const uint8_t *>(last.data()), repeat ? uint32_t(last.size()) : 0)) {
        return fail("the stored settings are invalid.");
    }
    trace("mode %d settings %u", repeat ? 1 : 0, unsigned(last.size()));

    int16 result = transfer(block, Dir::In);
    if (result == noErr) {
        int code = 0;
        if (!selb::run(STARLAB_COMMAND, {"--photoshop-bridge", block.name}, code, msg)) {
            result = fail(msg);
        } else {
            switch (selb::status(block)) {
                case selb::kStatusWritten:
                    result = transfer(block, Dir::Out);
                    if (result == noErr && !repeat && selb::settingsLength(block) <= selb::kSettingsCapacity) {
                        trace("remember settings %u", unsigned(selb::settingsLength(block)));
                        storeSettings(block.settings(), selb::settingsLength(block));
                    }
                    break;
                case selb::kStatusCancelled: result = userCanceledErr; break;
                default:
                    result = fail("Star Effects Lab closed without a result (exit code " + std::to_string(code) + ").");
            }
        }
    }
    setRects(rect(0, 0, 0, 0), rect(0, 0, 0, 0));
    return result;
}

// True while a Start runs (it pumps the host's run loop, so the host can re-enter us).
bool gBusy = false;
struct BusyScope {
    BusyScope() { gBusy = true; }
    ~BusyScope() { gBusy = false; }
};

}  // namespace

DLLExport MACPASCAL void PluginMain(const int16 selector, FilterRecordPtr filterRecord, intptr_t *data,
                                    int16 *result) {
    (void)data;
    try {
        *result = noErr;
        trace("selector %d", int(selector));
        if (selector == filterSelectorAbout) return;  // no dialog; the filter menu entry is enough
        if (gBusy) {  // re-entered during Start: leave the running one's globals alone
            *result = filterBadParameters;
            return;
        }
        gRec = filterRecord;
        gBig = gRec->bigDocumentData != nullptr;
        if (gBig) gRec->bigDocumentData->PluginUsing32BitCoordinates = true;
        switch (selector) {
            case filterSelectorPrepare:
                gRec->bufferSpace = 0;
                gRec->bufferSpace64 = 0;
                gRec->maxSpace64 = std::min<int64>(gRec->maxSpace64, kMaxHostSpace);
                gRec->maxSpace = int32(std::min<int64>(gRec->maxSpace, kMaxHostSpace));
                break;
            case filterSelectorStart: {
                BusyScope busy;
                *result = start();
                trace("start result %d", int(*result));
                break;
            }
            case filterSelectorParameters: gShowDialog = true; break;  // menu command, not Last Filter
            default: break;  // Continue/Finish: nothing to do (one-shot Start)
        }
    } catch (...) {
        trace("exception");
        *result = -1;
    }
}

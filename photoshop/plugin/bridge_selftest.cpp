// Proves C++ <-> Python protocol interop without Photoshop.
// usage: bridge_selftest <python-exe>   (python must import starlab)
#include <cstdio>
#include <cstring>

#include "bridge.h"

static const char *kApply =
    "import sys; from starlab.infrastructure.photoshop_bridge import PhotoshopBridge as B; "
    "b=B(sys.argv[1]); f=b.read(); b.write(None,f,f.pixels); b.close()";
// Interactive Apply that also remembers settings, like the GUI does.
static const char *kApplyRemember =
    "import sys; from starlab.infrastructure.photoshop_bridge import PhotoshopBridge as B; "
    "from starlab.domain.stars import DetectionSettings as D; from starlab.domain.effects.settings import EffectSettings as E; "
    "b=B(sys.argv[1]); assert b.mode==0; f=b.read(); b.store_settings(D(threshold_sigma=7.0), E()); b.write(None,f,f.pixels); b.close()";
// Repeat: reads the plug-in supplied settings and applies without a window.
static const char *kRepeat =
    "import sys; from starlab.infrastructure.photoshop_bridge import PhotoshopBridge as B; "
    "b=B(sys.argv[1]); assert b.mode==1; d,e=b.settings(); assert d.threshold_sigma==7.0; f=b.read(); b.write(None,f,f.pixels); b.close()";
static const char *kCancel =
    "import sys; from starlab.infrastructure.photoshop_bridge import PhotoshopBridge as B; "
    "b=B(sys.argv[1]); b.cancel(); b.close()";

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

// mode 0: apply (round trip), 1: cancel, 2: exit without touching the block
static void variant(const char *python, int mode, uint32_t expectStatus, bool expectPixels) {
    selb::Block b;
    std::string err;
    CHECK(selb::create(b, 64, 48, 3, 16, 32768, err));
    auto *px = reinterpret_cast<uint16_t *>(b.pixels());
    const size_t n = size_t(64) * 48 * 3;
    for (size_t i = 0; i < n; ++i) px[i] = uint16_t((i * 32768) / n);  // gradient, 0..32768
    std::vector<uint16_t> before(px, px + n);

    int code = -2;
    bool ok;
    if (mode == 2)
        ok = selb::run("/usr/bin/true", {}, code, err);
    else
        ok = selb::run(python, {"-c", mode == 0 ? kApply : kCancel, b.name}, code, err);
    CHECK(ok);
    std::printf("mode %d: exit %d status %u\n", mode, code, selb::status(b));
    CHECK(selb::status(b) == expectStatus);
    if (expectPixels) CHECK(std::memcmp(px, before.data(), n * 2) == 0);  // bit exact
    selb::destroy(b);
}

// Interactive apply stores settings; feeding them back as a repeat request must succeed and round trip pixels.
static void rememberAndRepeat(const char *python) {
    selb::Block a;
    std::string err;
    CHECK(selb::create(a, 64, 48, 3, 16, 32768, err));
    int code = -2;
    CHECK(selb::run(python, {"-c", kApplyRemember, a.name}, code, err));
    CHECK(code == 0 && selb::status(a) == selb::kStatusWritten);
    const uint32_t n = selb::settingsLength(a);
    std::printf("remembered %u bytes\n", n);
    CHECK(n > 0 && n <= selb::kSettingsCapacity);

    selb::Block b;
    CHECK(selb::create(b, 64, 48, 3, 16, 32768, err));
    CHECK(selb::setRequest(b, selb::kModeRepeat, a.settings(), n));
    auto *px = reinterpret_cast<uint16_t *>(b.pixels());
    const size_t count = size_t(64) * 48 * 3;
    for (size_t i = 0; i < count; ++i) px[i] = uint16_t((i * 32768) / count);
    std::vector<uint16_t> before(px, px + count);
    CHECK(selb::run(python, {"-c", kRepeat, b.name}, code, err));
    std::printf("repeat: exit %d status %u\n", code, selb::status(b));
    CHECK(code == 0 && selb::status(b) == selb::kStatusWritten);
    CHECK(std::memcmp(px, before.data(), count * 2) == 0);
    selb::destroy(a);
    selb::destroy(b);
}

int main(int argc, char **argv) {
    if (argc < 2) { std::printf("usage: bridge_selftest <python-exe>\n"); return 2; }
    variant(argv[1], 0, selb::kStatusWritten, true);
    variant(argv[1], 1, selb::kStatusCancelled, false);
    variant(argv[1], 2, selb::kStatusReady, false);
    rememberAndRepeat(argv[1]);
    // Missing executable must be reported, not crash.
    selb::Block b; std::string err; int code;
    CHECK(selb::create(b, 4, 4, 1, 8, 255, err));
    CHECK(!selb::run("/nonexistent/starlab", {}, code, err) && !err.empty());
    selb::destroy(b);
    std::printf(failures ? "SELFTEST FAILED\n" : "SELFTEST OK\n");
    return failures ? 1 : 0;
}

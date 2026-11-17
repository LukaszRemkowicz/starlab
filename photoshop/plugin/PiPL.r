// Binary PiPL for macOS Photoshop versions that predate PiPLs.json (e.g. Photoshop 2026).
// Keep in sync with PiPLs.json.
#include "PIGeneral.r"

resource 'PiPL' (16000, "Star Effects Lab", purgeable)
{
    {
        Kind { Filter },
        Name { "Star Effects Lab..." },
        Category { "Star Effects Lab" },
        Version { (latestFilterVersion << 16) | latestFilterSubVersion },
        CodeMacIntel64 { "PluginMain" },
        CodeMacARM64 { "PluginMain" },
        SupportedModes
        {
            noBitmap, doesSupportGrayScale, noIndexedColor, doesSupportRGBColor,
            noCMYKColor, noHSLColor, noHSBColor, noMultichannel, noDuotone, noLABColor
        },
        EnableInfo { "in (PSHOP_ImageMode, RGBMode, GrayScaleMode, RGB48Mode, Gray16Mode)" },
        PlugInMaxSize { 2000000, 2000000 },
        FilterCaseInfo
        {
            {
                inWhiteMat, outWhiteMat, doNotWriteOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination,
                inWhiteMat, outWhiteMat, writeOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination,
                inWhiteMat, outWhiteMat, writeOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination,
                inWhiteMat, outWhiteMat, doNotWriteOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination,
                inWhiteMat, outWhiteMat, writeOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination,
                inWhiteMat, outWhiteMat, doNotWriteOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination,
                inWhiteMat, outWhiteMat, writeOutsideSelection, filtersLayerMasks,
                    worksWithBlankData, copySourceToDestination
            }
        }
    }
};

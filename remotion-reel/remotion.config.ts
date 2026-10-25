/**
 * Note: When using the Node.JS APIs, the config file
 * doesn't apply. Instead, pass options directly to the APIs.
 *
 * All configuration options: https://remotion.dev/docs/config
 */

import { Config } from "@remotion/cli/config";

Config.setRspack(true);
Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
Config.setDefaultPremountInSeconds(2);
// WebGL for the light leaks and shader transitions: "angle" on macOS/Windows;
// Linux boxes often have no GPU-backed GL in headless Chrome, so use the
// software "swangle" there. REMOTION_GL overrides either.
Config.setChromiumOpenGlRenderer(
  (process.env.REMOTION_GL as "angle" | "swangle" | "egl" | undefined) ??
    (process.platform === "linux" ? "swangle" : "angle"),
);

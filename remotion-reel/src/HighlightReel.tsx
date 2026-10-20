import { lightLeak } from "@remotion/effects/light-leak";
import { Video } from "@remotion/media";
import { linearTiming, TransitionSeries, type TransitionPresentation } from "@remotion/transitions";
import { crossZoom } from "@remotion/transitions/cross-zoom";
import { fade } from "@remotion/transitions/fade";
import { filmBurn } from "@remotion/transitions/film-burn";
import { pushCut } from "@remotion/transitions/push-cut";
import { slide } from "@remotion/transitions/slide";
import { wipe } from "@remotion/transitions/wipe";
import { zoomBlur } from "@remotion/transitions/zoom-blur";
import React from "react";
import {
  AbsoluteFill,
  Easing,
  Freeze,
  interpolate,
  Sequence,
  Solid,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const FPS = 30;
const TITLE = 3 * FPS;
const END = Math.round(2.5 * FPS);

type Effect = "plain" | "freeze" | "punch_zoom" | "speed_ramp" | "replay" | "zoom_replay";
// make: clip-local seconds of the make; pre/post: seconds kept around it;
// rimX/rimY: rim centre in % of the frame, where zooms aim.
type Clip = { file: string; make: number; pre: number; post: number; effect: Effect; rimX: number; rimY: number };

// Written by shot-clipper-highlights (src/shot_clipper/highlights.py) and
// passed with `npx remotion render HighlightReel out.mp4 --props=<file>`.
export type ReelProps = {
  titleLines: string[];
  clips: Clip[];
  transitionFrames: number;
  leakEvery: number; // every Nth cut is a light leak instead of a transition; 0 = never
  presentations: string[];
  seed: number;
};

// One piece of a clip: source seconds [from, to) at `rate`, optionally zoomed
// into the rim or held as a freeze frame for `hold` seconds afterwards.
type Part = { from: number; to: number; rate: number; zoom?: boolean; hold?: number };

const partsFor = (c: Clip): Part[] => {
  const a = Math.max(0, c.make - c.pre);
  const b = c.make + c.post;
  const slowA = Math.max(a, c.make - 0.9);
  const slowB = c.make + 0.6;
  switch (c.effect) {
    case "freeze":
      return [
        { from: a, to: c.make + 0.15, rate: 1, hold: 0.7 },
        { from: c.make + 0.15, to: b, rate: 1 },
      ];
    case "punch_zoom":
      return [
        { from: a, to: Math.max(a, c.make - 0.7), rate: 1 },
        { from: Math.max(a, c.make - 0.7), to: b, rate: 1, zoom: true },
      ];
    case "speed_ramp":
      return [
        { from: a, to: slowA, rate: 1 },
        { from: slowA, to: slowB, rate: 0.4 },
        { from: slowB, to: b, rate: 1 },
      ];
    case "replay":
    case "zoom_replay":
      return [
        { from: a, to: b, rate: 1 },
        { from: slowA, to: slowB, rate: 0.45, zoom: c.effect === "zoom_replay" },
      ];
    default:
      return [{ from: a, to: b, rate: 1 }];
  }
};

const playFrames = (p: Part) => Math.max(1, Math.round(((p.to - p.from) / p.rate) * FPS));
const holdFrames = (p: Part) => Math.round((p.hold ?? 0) * FPS);
export const clipFrames = (c: Clip) =>
  partsFor(c).reduce((sum, p) => sum + playFrames(p) + holdFrames(p), 0);

// Zooms toward the rim over the part (punch zoom eases in fast, replay zoom slowly).
const RimZoom: React.FC<{ clip: Clip; max: number; children: React.ReactNode }> = ({
  clip,
  max,
  children,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const scale = interpolate(frame, [0, Math.min(durationInFrames - 1, 12)], [1, max], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  return (
    <AbsoluteFill
      style={{ transform: `scale(${scale})`, transformOrigin: `${clip.rimX}% ${clip.rimY}%` }}
    >
      {children}
    </AbsoluteFill>
  );
};

// White flash + slight push-in on a freeze frame.
const FreezeFlash: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const flash = interpolate(frame, [0, 5], [0.7, 0], { extrapolateRight: "clamp" });
  const scale = interpolate(frame, [0, durationInFrames], [1, 1.06]);
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{ transform: `scale(${scale})`, filter: "saturate(1.25) contrast(1.08)" }}>
        {children}
      </AbsoluteFill>
      <AbsoluteFill style={{ backgroundColor: "white", opacity: flash }} />
    </AbsoluteFill>
  );
};

const PartVideo: React.FC<{ clip: Clip; part: Part; muted?: boolean }> = ({ clip, part, muted }) => (
  <Video
    src={staticFile(clip.file)}
    trimBefore={Math.round(part.from * FPS)}
    playbackRate={part.rate}
    muted={muted || part.rate !== 1}
    objectFit="cover"
    style={{ width: "100%", height: "100%" }}
  />
);

const ClipScene: React.FC<{ clip: Clip }> = ({ clip }) => {
  let at = 0;
  const nodes: React.ReactNode[] = [];
  partsFor(clip).forEach((part, i) => {
    const len = playFrames(part);
    const video = <PartVideo clip={clip} part={part} />;
    nodes.push(
      <Sequence key={`p${i}`} from={at} durationInFrames={len} name={`${part.rate < 1 ? "slow" : "play"} ${i}`}>
        {part.zoom ? (
          <RimZoom clip={clip} max={part.rate < 1 ? 2.2 : 1.9}>
            {video}
          </RimZoom>
        ) : (
          video
        )}
      </Sequence>,
    );
    at += len;
    if (part.hold) {
      const hold = holdFrames(part);
      nodes.push(
        <Sequence key={`h${i}`} from={at} durationInFrames={hold} name="freeze">
          <FreezeFlash>
            <Freeze frame={len - 1}>
              <PartVideo clip={clip} part={part} muted />
            </Freeze>
          </FreezeFlash>
        </Sequence>,
      );
      at += hold;
    }
  });
  return <AbsoluteFill style={{ backgroundColor: "black", overflow: "hidden" }}>{nodes}</AbsoluteFill>;
};

const Card: React.FC<{ lines: { text: string; size: number; color: string }[] }> = ({ lines }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill
      style={{
        background: "radial-gradient(ellipse at 50% 40%, #2a1b0e 0%, #0b0b10 70%)",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Helvetica Neue, Arial, sans-serif",
        fontWeight: 800,
      }}
    >
      {lines.map((l, i) => {
        const s = spring({ frame: frame - i * 6, fps, config: { damping: 14 } });
        return (
          <div
            key={l.text}
            style={{
              fontSize: l.size,
              color: l.color,
              letterSpacing: 4,
              margin: 10,
              opacity: s,
              transform: `translateY(${(1 - s) * 40}px) scale(${0.9 + 0.1 * s})`,
            }}
          >
            {l.text}
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

const LightLeakOverlay: React.FC<{ seed: number; hueShift: number }> = ({ seed, hueShift }) => {
  const frame = useCurrentFrame();
  const { durationInFrames, height, width } = useVideoConfig();
  return (
    <Solid
      width={width}
      height={height}
      effects={[
        lightLeak({
          seed,
          hueShift,
          progress: interpolate(frame, [0, durationInFrames - 1], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }),
      ]}
    />
  );
};

// Deterministic pseudo-random so Studio preview and render pick the same cuts.
const rand = (i: number) => {
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const PRESENTATIONS: Record<string, () => TransitionPresentation<any>> = {
  fade: () => fade(),
  slideRight: () => slide({ direction: "from-right" }),
  slideLeft: () => slide({ direction: "from-left" }),
  wipe: () => wipe({ direction: "from-left" }),
  pushCut: () => pushCut({}),
  crossZoom: () => crossZoom({}),
  zoomBlur: () => zoomBlur({}),
  filmBurn: () => filmBurn({ seed: 3 }),
};

const isLeakCut = (cut: number, every: number) => every > 0 && cut % every === every - 2;

export const reelFrames = (p: ReelProps) => {
  const scenes = TITLE + END + p.clips.reduce((s, c) => s + clipFrames(c), 0);
  let overlap = 0;
  for (let cut = 0; cut <= p.clips.length; cut++) if (!isLeakCut(cut, p.leakEvery)) overlap += p.transitionFrames;
  return scenes - overlap;
};

export const HighlightReel: React.FC<ReelProps> = (p) => {
  const { width, height } = useVideoConfig();
  const colors = ["#ffffff", "#ffc440", "#c8cdd7"];
  const sizes = [120, 56, 44];
  const scenes: { name: string; frames: number; node: React.ReactNode }[] = [
    {
      name: "Title",
      frames: TITLE,
      node: <Card lines={p.titleLines.map((text, i) => ({ text, size: sizes[i] ?? 40, color: colors[i] ?? "#fff" }))} />,
    },
    ...p.clips.map((c) => ({
      name: `${c.file.split("/").pop()?.replace(".mp4", "")} · ${c.effect}`,
      frames: clipFrames(c),
      node: <ClipScene clip={c} />,
    })),
    { name: "End", frames: END, node: <Card lines={[{ text: "🏀", size: 140, color: "#ff8c28" }]} /> },
  ];
  const names = p.presentations.filter((n) => n in PRESENTATIONS);
  const pool = names.length ? names : ["fade"];

  const children: React.ReactNode[] = [];
  scenes.forEach((s, i) => {
    if (i > 0) {
      const cut = i - 1;
      if (isLeakCut(cut, p.leakEvery)) {
        children.push(
          <TransitionSeries.Overlay key={`o${i}`} durationInFrames={24}>
            <LightLeakOverlay seed={cut + p.seed} hueShift={Math.round(rand(cut + p.seed) * 60)} />
          </TransitionSeries.Overlay>,
        );
      } else {
        const name = i === 1 ? "fade" : pool[Math.floor(rand(cut + 100 + p.seed * 7) * pool.length)];
        children.push(
          <TransitionSeries.Transition
            key={`t${i}`}
            presentation={(PRESENTATIONS[name] ?? PRESENTATIONS.fade)()}
            timing={linearTiming({ durationInFrames: p.transitionFrames })}
          />,
        );
      }
    }
    children.push(
      <TransitionSeries.Sequence key={`s${i}`} name={s.name} durationInFrames={s.frames}>
        {s.node}
      </TransitionSeries.Sequence>,
    );
  });

  return (
    <AbsoluteFill style={{ backgroundColor: "black", width, height }}>
      <TransitionSeries>{children}</TransitionSeries>
    </AbsoluteFill>
  );
};

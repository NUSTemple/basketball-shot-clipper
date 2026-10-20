import { CalculateMetadataFunction, Composition } from "remotion";
import { FPS, HighlightReel, ReelProps, reelFrames } from "./HighlightReel";

// Studio preview without props: an empty reel (title + end card). Real reels
// pass their clips with --props, see src/shot_clipper/highlights.py.
const defaultProps: ReelProps = {
  titleLines: ["HIGHLIGHTS"],
  clips: [],
  transitionFrames: 14,
  leakEvery: 5,
  presentations: ["fade"],
  seed: 1,
};

const calculateMetadata: CalculateMetadataFunction<ReelProps> = ({ props }) => ({
  durationInFrames: reelFrames(props),
});

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="HighlightReel"
      component={HighlightReel}
      durationInFrames={reelFrames(defaultProps)}
      fps={FPS}
      width={1920}
      height={1080}
      defaultProps={defaultProps}
      calculateMetadata={calculateMetadata}
    />
  );
};

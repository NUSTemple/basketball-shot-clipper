import { forwardRef, useImperativeHandle, useRef } from 'react'

export interface VideoPlayerHandle {
  seek: (seconds: number) => void
}

interface VideoPlayerProps {
  src: string
  onTimeUpdate: (seconds: number) => void
  onDuration: (seconds: number) => void
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(function VideoPlayer(
  { src, onTimeUpdate, onDuration },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null)

  useImperativeHandle(ref, () => ({
    seek(seconds: number) {
      const el = videoRef.current
      if (!el) return
      el.currentTime = seconds
      void el.play().catch(() => {
        /* autoplay can be blocked - the user still sees the seek happen */
      })
    },
  }))

  return (
    <video
      ref={videoRef}
      src={src}
      controls
      className="aspect-video w-full rounded-md bg-black"
      onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
      onLoadedMetadata={(e) => onDuration(e.currentTarget.duration)}
    />
  )
})

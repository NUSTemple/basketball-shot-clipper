import { useEffect, useRef, useState } from 'react'

interface HoopCalibrationCanvasProps {
  imageSrc: string
  onBoxChange: (bbox: [number, number, number, number], frameWidth: number, frameHeight: number) => void
}

interface DragState {
  startX: number
  startY: number
  endX: number
  endY: number
}

// A box small enough to be a stray click rather than a real drag (fraction
// of image width/height) - guards against onBoxChange firing with a
// degenerate, effectively-zero-area box that would silently enable "Save"
// on nothing meaningful.
const MIN_BOX_FRACTION = 0.01

// Drag-to-draw a box over the still image, in image-relative fractions
// (0..1) - this is exactly hoop_bbox_norm's own coordinate space, so no
// pixel<->fraction conversion is needed anywhere else in the app.
//
// Listens on window, not just the container, once a drag starts: a
// container-scoped mousemove/mouseup pair stops tracking the instant the
// cursor crosses outside the container's (often small, aspect-video-sized)
// bounds mid-drag - a classic drag-implementation bug where fast or
// edge-reaching drags just silently stop updating.
export function HoopCalibrationCanvas({ imageSrc, onBoxChange }: HoopCalibrationCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 })
  const [drag, setDrag] = useState<DragState | null>(null)
  // Mirrors `drag` so the window mouseup handler can read the final box
  // without calling onBoxChange from inside a setState updater (updaters
  // must be pure - React runs them twice in StrictMode).
  const dragRef = useRef<DragState | null>(null)
  const updateDrag = (next: DragState | null) => {
    dragRef.current = next
    setDrag(next)
  }

  const fractionFromPoint = (clientX: number, clientY: number) => {
    const rect = containerRef.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (clientY - rect.top) / rect.height)),
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault() // stops the browser's native "drag this image" gesture from hijacking the interaction
    const { x, y } = fractionFromPoint(e.clientX, e.clientY)
    updateDrag({ startX: x, startY: y, endX: x, endY: y })
  }

  useEffect(() => {
    if (!drag) return

    const handleMove = (e: MouseEvent) => {
      const prev = dragRef.current
      if (!prev) return
      const { x, y } = fractionFromPoint(e.clientX, e.clientY)
      updateDrag({ ...prev, endX: x, endY: y })
    }
    const handleUp = () => {
      const final = dragRef.current
      if (final) {
        const width = Math.abs(final.endX - final.startX)
        const height = Math.abs(final.endY - final.startY)
        if (width >= MIN_BOX_FRACTION && height >= MIN_BOX_FRACTION) {
          onBoxChange(
            [
              Math.min(final.startX, final.endX),
              Math.min(final.startY, final.endY),
              Math.max(final.startX, final.endX),
              Math.max(final.startY, final.endY),
            ],
            naturalSize.width,
            naturalSize.height,
          )
        }
      }
      updateDrag(null)
    }

    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
    return () => {
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
    }
    // drag is intentionally the only dependency that re-subscribes - only
    // its start (non-null) / end (null) transitions should attach/detach
    // the window listeners, not every coordinate update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drag !== null])

  const boxStyle = drag
    ? {
        left: `${Math.min(drag.startX, drag.endX) * 100}%`,
        top: `${Math.min(drag.startY, drag.endY) * 100}%`,
        width: `${Math.abs(drag.endX - drag.startX) * 100}%`,
        height: `${Math.abs(drag.endY - drag.startY) * 100}%`,
      }
    : undefined

  return (
    <div>
      <div
        ref={containerRef}
        className="relative w-full cursor-crosshair select-none"
        onMouseDown={handleMouseDown}
      >
        <img
          src={imageSrc}
          alt="Calibration frame"
          className="w-full select-none [-webkit-user-drag:none]"
          draggable={false}
          onLoad={(e) =>
            setNaturalSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })
          }
        />
        {boxStyle && (
          <div className="pointer-events-none absolute border-2 border-orange-500 bg-orange-500/20" style={boxStyle} />
        )}
      </div>
      <p className="mt-1 text-xs text-slate-500">Drag a box around the basket.</p>
    </div>
  )
}

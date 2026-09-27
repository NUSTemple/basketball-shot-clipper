import { useRef, useState } from 'react'

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

// Drag-to-draw a box over the still image, in image-relative fractions
// (0..1) - this is exactly hoop_bbox_norm's own coordinate space, so no
// pixel<->fraction conversion is needed anywhere else in the app.
export function HoopCalibrationCanvas({ imageSrc, onBoxChange }: HoopCalibrationCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 })
  const [drag, setDrag] = useState<DragState | null>(null)

  const fractionFromEvent = (e: React.MouseEvent) => {
    const rect = containerRef.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    const { x, y } = fractionFromEvent(e)
    setDrag({ startX: x, startY: y, endX: x, endY: y })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drag) return
    const { x, y } = fractionFromEvent(e)
    setDrag({ ...drag, endX: x, endY: y })
  }

  const handleMouseUp = () => {
    if (!drag) return
    const bbox: [number, number, number, number] = [
      Math.min(drag.startX, drag.endX),
      Math.min(drag.startY, drag.endY),
      Math.max(drag.startX, drag.endX),
      Math.max(drag.startY, drag.endY),
    ]
    onBoxChange(bbox, naturalSize.width, naturalSize.height)
  }

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
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <img
          src={imageSrc}
          alt="Calibration frame"
          className="w-full select-none"
          draggable={false}
          onLoad={(e) =>
            setNaturalSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })
          }
        />
        {boxStyle && (
          <div className="pointer-events-none absolute border-2 border-orange-500 bg-orange-500/20" style={boxStyle} />
        )}
      </div>
      <p className="mt-1 text-xs text-slate-500">Drag a box around the hoop.</p>
    </div>
  )
}

import { useRef, useState } from 'react'

interface UploadDropzoneProps {
  onFileSelected: (file: File) => void
  disabled?: boolean
}

export function UploadDropzone({ onFileSelected, disabled }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setIsDragOver(true)
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setIsDragOver(false)
        const file = e.dataTransfer.files[0]
        if (file && !disabled) onFileSelected(file)
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`flex aspect-video cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed text-sm ${
        disabled
          ? 'cursor-not-allowed border-slate-200 text-slate-300'
          : isDragOver
            ? 'border-orange-500 bg-orange-50 text-orange-700'
            : 'border-slate-300 text-slate-500 hover:border-slate-400'
      }`}
    >
      <p>Drag a video here, or click to choose a file</p>
      <p className="mt-1 text-xs text-slate-400">.mp4 or .mov</p>
      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.mov,video/mp4,video/quicktime"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFileSelected(file)
          e.target.value = ''
        }}
      />
    </div>
  )
}

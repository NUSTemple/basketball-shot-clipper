import { useRef, useState } from 'react'

interface UploadDropzoneProps {
  onFilesSelected: (files: File[]) => void
  disabled?: boolean
}

export function UploadDropzone({ onFilesSelected, disabled }: UploadDropzoneProps) {
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
        const files = Array.from(e.dataTransfer.files)
        if (files.length > 0 && !disabled) onFilesSelected(files)
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`flex h-40 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed text-sm ${
        disabled
          ? 'cursor-not-allowed border-slate-200 text-slate-300'
          : isDragOver
            ? 'border-orange-500 bg-orange-50 text-orange-700'
            : 'border-slate-300 text-slate-500 hover:border-slate-400'
      }`}
    >
      <p>Drag videos here, or click to choose files</p>
      <p className="mt-1 text-xs text-slate-400">.mp4 or .mov - multiple files supported</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".mp4,.mov,video/mp4,video/quicktime"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          if (files.length > 0) onFilesSelected(files)
          e.target.value = ''
        }}
      />
    </div>
  )
}

import { useEffect, useState, useRef, useCallback } from 'react'
import { api } from '../api'
import { X, Save, Trash2, ChevronLeft, ChevronRight } from 'lucide-react'

export interface AnnotationBox {
  class_id: number
  center_x: number
  center_y: number
  width: number
  height: number
}

interface Props {
  samples: { sample_id: string; filename: string; dataset_name: string; file_path: string; annotated: boolean }[]
  initialIndex: number
  classes: string[]
  onClose: () => void
  onSaved: () => void
}

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316']

export function AnnotationEditor({ samples, initialIndex, classes, onClose, onSaved }: Props) {
  const [idx, setIdx] = useState(initialIndex)
  const [boxes, setBoxes] = useState<AnnotationBox[]>([])
  const [selected, setSelected] = useState(-1)
  const [drawClassId, setDrawClassId] = useState(0)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 })
  const [drawing, setDrawing] = useState<{ x: number; y: number } | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const sample = samples[idx]
  // file_path is like "samples\warehouse_v1\xxx.png" — serve from /samples/
  const imgUrl = sample ? `/samples/${sample.file_path.replace(/^samples[\\/]/, '').replace(/\\/g, '/')}` : ''

  // Load annotations when sample changes
  useEffect(() => {
    if (!sample) return
    api.getAnnotations(sample.sample_id).then(d => {
      setBoxes((d.annotations || []) as unknown as AnnotationBox[])
      setDirty(false)
      setSelected(-1)
    }).catch(() => { setBoxes([]); setDirty(false) })
  }, [sample])

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !imgSize.w) return
    const ctx = canvas.getContext('2d')!
    const cw = canvas.width
    const ch = canvas.height

    ctx.clearRect(0, 0, cw, ch)
    ctx.drawImage(img, 0, 0, cw, ch)

    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i]
      const x = (b.center_x - b.width / 2) * cw
      const y = (b.center_y - b.height / 2) * ch
      const w = b.width * cw
      const h = b.height * ch
      const color = COLORS[b.class_id % COLORS.length]
      const isSel = i === selected

      ctx.strokeStyle = color
      ctx.lineWidth = isSel ? 2.5 : 1.5
      ctx.strokeRect(x, y, w, h)

      // Fill
      ctx.fillStyle = color + (isSel ? '30' : '15')
      ctx.fillRect(x, y, w, h)

      // Label
      const label = `${classes[b.class_id] || b.class_id}`
      ctx.font = '12px Fira Sans, sans-serif'
      const tw = ctx.measureText(label).width + 8
      ctx.fillStyle = color
      ctx.fillRect(x, y - 18, tw, 18)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + 4, y - 5)

      // Resize handles for selected
      if (isSel) {
        const hs = 5
        ctx.fillStyle = color
        for (const [hx, hy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
          ctx.fillRect(hx - hs, hy - hs, hs * 2, hs * 2)
        }
      }
    }

    // Drawing preview
    if (drawing) {
      ctx.setLineDash([4, 4])
      ctx.strokeStyle = COLORS[drawClassId % COLORS.length]
      ctx.lineWidth = 1.5
      // Will be drawn in mousemove via tempRect
    }
    ctx.setLineDash([])
  }, [boxes, selected, imgSize, classes, drawing, drawClassId])

  useEffect(() => { draw() }, [draw])

  const onImgLoad = () => {
    const img = imgRef.current!
    setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
    if (canvasRef.current && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      canvasRef.current.width = rect.width
      canvasRef.current.height = rect.height
    }
  }

  // Resize canvas on container resize
  useEffect(() => {
    const obs = new ResizeObserver(() => {
      if (canvasRef.current && containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        canvasRef.current.width = rect.width
        canvasRef.current.height = rect.height
        draw()
      }
    })
    if (containerRef.current) obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [draw])

  // Mouse state for drawing/dragging
  const dragRef = useRef<{
    mode: 'draw' | 'move' | 'resize'
    startX: number; startY: number
    boxIdx: number; corner: number
    origBox: AnnotationBox
    curX: number; curY: number
  } | null>(null)

  const canvasXY = (e: React.MouseEvent) => {
    const c = canvasRef.current!
    const r = c.getBoundingClientRect()
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height }
  }

  const hitTest = (mx: number, my: number): { boxIdx: number; corner: number } => {
    const c = canvasRef.current!
    const cw = c.width, ch = c.height
    // Check corners first (for resize), then box body (for move)
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i]
      const x1 = (b.center_x - b.width / 2) * cw
      const y1 = (b.center_y - b.height / 2) * ch
      const x2 = x1 + b.width * cw
      const y2 = y1 + b.height * ch
      const px = mx * cw, py = my * ch
      const ht = 8
      const corners = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]
      for (let ci = 0; ci < 4; ci++) {
        if (Math.abs(px - corners[ci][0]) < ht && Math.abs(py - corners[ci][1]) < ht) {
          return { boxIdx: i, corner: ci }
        }
      }
      if (px >= x1 && px <= x2 && py >= y1 && py <= y2) {
        return { boxIdx: i, corner: -1 }
      }
    }
    return { boxIdx: -1, corner: -1 }
  }

  const onMouseDown = (e: React.MouseEvent) => {
    const { x, y } = canvasXY(e)
    const hit = hitTest(x, y)
    if (hit.boxIdx >= 0) {
      setSelected(hit.boxIdx)
      dragRef.current = {
        mode: hit.corner >= 0 ? 'resize' : 'move',
        startX: x, startY: y,
        boxIdx: hit.boxIdx, corner: hit.corner,
        origBox: { ...boxes[hit.boxIdx] },
        curX: x, curY: y,
      }
    } else {
      setSelected(-1)
      // Start drawing new box
      dragRef.current = {
        mode: 'draw',
        startX: x, startY: y,
        boxIdx: -1, corner: -1,
        origBox: { class_id: drawClassId, center_x: 0, center_y: 0, width: 0, height: 0 },
        curX: x, curY: y,
      }
      setDrawing({ x, y })
    }
  }

  const onMouseMove = (e: React.MouseEvent) => {
    const d = dragRef.current
    if (!d) return
    const { x, y } = canvasXY(e)
    d.curX = x; d.curY = y

    if (d.mode === 'draw') {
      // Preview rect
      const canvas = canvasRef.current!
      const ctx = canvas.getContext('2d')!
      draw()
      const x1 = Math.min(d.startX, x) * canvas.width
      const y1 = Math.min(d.startY, y) * canvas.height
      const w = Math.abs(x - d.startX) * canvas.width
      const h = Math.abs(y - d.startY) * canvas.height
      ctx.setLineDash([4, 4])
      ctx.strokeStyle = COLORS[drawClassId % COLORS.length]
      ctx.lineWidth = 1.5
      ctx.strokeRect(x1, y1, w, h)
      ctx.setLineDash([])
    } else if (d.mode === 'move') {
      const dx = x - d.startX
      const dy = y - d.startY
      const updated = [...boxes]
      updated[d.boxIdx] = {
        ...d.origBox,
        center_x: clamp(d.origBox.center_x + dx, 0, 1),
        center_y: clamp(d.origBox.center_y + dy, 0, 1),
      }
      setBoxes(updated)
      setDirty(true)
    } else if (d.mode === 'resize') {
      const ob = d.origBox
      const ox1 = ob.center_x - ob.width / 2
      const oy1 = ob.center_y - ob.height / 2
      const ox2 = ob.center_x + ob.width / 2
      const oy2 = ob.center_y + ob.height / 2
      let nx1 = ox1, ny1 = oy1, nx2 = ox2, ny2 = oy2
      if (d.corner === 0) { nx1 = x; ny1 = y }
      else if (d.corner === 1) { nx2 = x; ny1 = y }
      else if (d.corner === 2) { nx1 = x; ny2 = y }
      else if (d.corner === 3) { nx2 = x; ny2 = y }
      if (nx1 > nx2) [nx1, nx2] = [nx2, nx1]
      if (ny1 > ny2) [ny1, ny2] = [ny2, ny1]
      nx1 = clamp(nx1, 0, 1); nx2 = clamp(nx2, 0, 1)
      ny1 = clamp(ny1, 0, 1); ny2 = clamp(ny2, 0, 1)
      const updated = [...boxes]
      updated[d.boxIdx] = {
        ...ob,
        center_x: (nx1 + nx2) / 2,
        center_y: (ny1 + ny2) / 2,
        width: nx2 - nx1,
        height: ny2 - ny1,
      }
      setBoxes(updated)
      setDirty(true)
    }
  }

  const onMouseUp = () => {
    const d = dragRef.current
    if (d && d.mode === 'draw') {
      const x1 = Math.min(d.startX, d.curX)
      const y1 = Math.min(d.startY, d.curY)
      const x2 = Math.max(d.startX, d.curX)
      const y2 = Math.max(d.startY, d.curY)
      const w = x2 - x1, h = y2 - y1
      if (w > 0.01 && h > 0.01) {
        const newBox: AnnotationBox = {
          class_id: drawClassId,
          center_x: (x1 + x2) / 2,
          center_y: (y1 + y2) / 2,
          width: w,
          height: h,
        }
        setBoxes(prev => [...prev, newBox])
        setSelected(boxes.length)
        setDirty(true)
      }
      setDrawing(null)
    }
    dragRef.current = null
  }

  const deleteSelected = () => {
    if (selected < 0) return
    setBoxes(prev => prev.filter((_, i) => i !== selected))
    setSelected(-1)
    setDirty(true)
  }

  const changeClass = (newCls: number) => {
    if (selected < 0) return
    const updated = [...boxes]
    updated[selected] = { ...updated[selected], class_id: newCls }
    setBoxes(updated)
    setDirty(true)
  }

  const save = async () => {
    if (!sample) return
    setSaving(true)
    try {
      await api.saveAnnotations(sample.sample_id, {
        annotations: boxes,
        dataset_classes: classes,
      })
      setDirty(false)
      onSaved()
    } catch (e) { alert((e as Error).message) }
    finally { setSaving(false) }
  }

  const nav = async (dir: -1 | 1) => {
    if (dirty && !confirm('标注未保存，确定切换？')) return
    setIdx(i => Math.max(0, Math.min(samples.length - 1, i + dir)))
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') { deleteSelected(); e.preventDefault() }
      if (e.key === 'ArrowLeft') { nav(-1); e.preventDefault() }
      if (e.key === 'ArrowRight') { nav(1); e.preventDefault() }
      if (e.key === 's' && (e.ctrlKey || e.metaKey)) { save(); e.preventDefault() }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  if (!sample) return null

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex flex-col" onClick={onClose}>
      <div className="flex-1 flex min-h-0" onClick={e => e.stopPropagation()}>
        {/* Left: Canvas */}
        <div className="flex-1 flex flex-col min-w-0 p-3 gap-2">
          {/* Toolbar */}
          <div className="flex items-center gap-2 shrink-0">
            <button onClick={() => nav(-1)} disabled={idx <= 0} className="p-1.5 rounded-lg bg-card border border-border text-muted hover:text-foreground disabled:opacity-20 cursor-pointer transition-colors"><ChevronLeft size={15} /></button>
            <span className="text-xs text-muted font-mono">{idx + 1} / {samples.length}</span>
            <button onClick={() => nav(1)} disabled={idx >= samples.length - 1} className="p-1.5 rounded-lg bg-card border border-border text-muted hover:text-foreground disabled:opacity-20 cursor-pointer transition-colors"><ChevronRight size={15} /></button>
            <span className="text-xs text-muted truncate max-w-[200px]">{sample.filename}</span>
            {sample.annotated && <span className="text-[10px] text-success bg-success-dim px-1.5 py-0.5 rounded">已标注</span>}
            <div className="ml-auto flex items-center gap-2">
              {dirty && <span className="text-[10px] text-warning">● 未保存</span>}
              <button onClick={save} disabled={saving || !dirty} className="flex items-center gap-1.5 px-3 py-1.5 bg-accent text-white rounded-lg text-xs cursor-pointer hover:bg-accent-light transition-colors disabled:opacity-30">
                <Save size={12} /> {saving ? '保存中…' : '保存'}
              </button>
              <button onClick={onClose} className="p-1.5 rounded-lg bg-card border border-border text-muted hover:text-foreground cursor-pointer transition-colors"><X size={15} /></button>
            </div>
          </div>

          {/* Canvas area */}
          <div ref={containerRef} className="flex-1 relative bg-black rounded-xl overflow-hidden min-h-0">
            <img
              ref={imgRef}
              src={imgUrl}
              alt=""
              onLoad={onImgLoad}
              className="absolute inset-0 w-full h-full object-contain opacity-0"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full cursor-crosshair"
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseUp}
            />
          </div>
        </div>

        {/* Right: Panel */}
        <div className="w-56 bg-surface border-l border-border flex flex-col shrink-0">
          {/* Draw class selector */}
          <div className="p-3 border-b border-border">
            <div className="text-[10px] text-muted uppercase tracking-wider mb-2">绘制类别</div>
            <div className="flex flex-col gap-1">
              {classes.map((cls, ci) => (
                <button
                  key={ci}
                  onClick={() => setDrawClassId(ci)}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs cursor-pointer transition-colors ${drawClassId === ci ? 'bg-accent-dim text-accent font-medium' : 'text-muted hover:text-foreground hover:bg-card'}`}
                >
                  <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: COLORS[ci % COLORS.length] }} />
                  {cls}
                </button>
              ))}
            </div>
          </div>

          {/* Box list */}
          <div className="flex-1 overflow-y-auto p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-muted uppercase tracking-wider">标注框 ({boxes.length})</span>
            </div>
            {boxes.length === 0 ? (
              <p className="text-xs text-muted/50 mt-4 text-center">在图片上拖拽绘制标注框</p>
            ) : (
              <div className="flex flex-col gap-1">
                {boxes.map((b, i) => (
                  <div
                    key={i}
                    onClick={() => setSelected(i)}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs cursor-pointer transition-colors ${selected === i ? 'bg-card border border-accent/40' : 'hover:bg-card border border-transparent'}`}
                  >
                    <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: COLORS[b.class_id % COLORS.length] }} />
                    <select
                      value={b.class_id}
                      onChange={e => { setSelected(i); changeClass(Number(e.target.value)) }}
                      onClick={e => e.stopPropagation()}
                      className="bg-transparent text-xs flex-1 min-w-0 cursor-pointer focus:outline-none"
                    >
                      {classes.map((cls, ci) => <option key={ci} value={ci}>{cls}</option>)}
                    </select>
                    <button
                      onClick={e => { e.stopPropagation(); setSelected(i); setTimeout(deleteSelected, 0) }}
                      className="p-0.5 rounded text-muted hover:text-danger cursor-pointer transition-colors opacity-0 group-hover:opacity-100"
                      style={{ opacity: selected === i ? 1 : undefined }}
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Keyboard hints */}
          <div className="p-3 border-t border-border text-[10px] text-muted/50 space-y-0.5">
            <div>← → 切换样本</div>
            <div>Delete 删除选中框</div>
            <div>Ctrl+S 保存</div>
          </div>
        </div>
      </div>
    </div>
  )
}

function clamp(v: number, min: number, max: number) { return Math.max(min, Math.min(max, v)) }

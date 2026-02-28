import { useRef, useEffect, useState, useCallback } from 'react'
import { Trash2, Plus, MousePointer, Minus, RefreshCw } from 'lucide-react'

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */
export type Point = [number, number]

export interface Tripwire {
  id: string
  name: string
  p1: Point
  p2: Point
  direction: string
  cooldown?: number
}

interface Props {
  cameraId: string
  width: number
  height: number
  roi: Point[]
  tripwires: Tripwire[]
  onRoiChange: (roi: Point[]) => void
  onTripwiresChange: (tws: Tripwire[]) => void
}

type Mode = 'select' | 'roi' | 'tripwire'

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export function DrawingCanvas({ cameraId, width, height, roi, tripwires, onRoiChange, onTripwiresChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [mode, setMode] = useState<Mode>('select')
  const [dragIdx, setDragIdx] = useState<{ type: 'roi' | 'tw'; twIdx?: number; ptKey?: 'p1' | 'p2'; idx: number } | null>(null)
  const [twDraft, setTwDraft] = useState<Point | null>(null) // first point of new tripwire
  const [mousePos, setMousePos] = useState<Point | null>(null)
  const [scale, setScale] = useState(1)
  const [imgVer, setImgVer] = useState(0)

  // Load snapshot
  const loadSnapshot = useCallback(() => {
    if (!cameraId) return
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => { imgRef.current = img; setImgVer(v => v + 1) }
    img.onerror = () => { imgRef.current = null; setImgVer(v => v + 1) }
    img.src = `/snapshot/${cameraId}?t=${Date.now()}`
  }, [cameraId])

  useEffect(() => { loadSnapshot() }, [loadSnapshot])

  // Compute scale (with ResizeObserver)
  useEffect(() => {
    if (!containerRef.current) return
    const update = () => {
      if (containerRef.current) setScale(containerRef.current.clientWidth / width)
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [width])

  const toCanvas = useCallback((p: Point): [number, number] => [p[0] * scale, p[1] * scale], [scale])
  const toReal = useCallback((cx: number, cy: number): Point => [Math.round(cx / scale), Math.round(cy / scale)], [scale])

  const getCanvasXY = useCallback((e: React.MouseEvent): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }, [])

  // ---- Draw ----
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const cw = width * scale
    const ch = height * scale
    canvas.width = cw
    canvas.height = ch
    ctx.clearRect(0, 0, cw, ch)

    // Background image
    if (imgRef.current) {
      ctx.drawImage(imgRef.current, 0, 0, cw, ch)
    } else {
      ctx.fillStyle = '#0a0f1a'
      ctx.fillRect(0, 0, cw, ch)
      ctx.fillStyle = '#475569'
      ctx.font = '14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('无画面 — 请确认摄像头在线', cw / 2, ch / 2)
    }

    // ROI polygon
    if (roi.length > 0) {
      ctx.beginPath()
      const pts = roi.map(toCanvas)
      ctx.moveTo(pts[0][0], pts[0][1])
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
      ctx.closePath()
      ctx.fillStyle = 'rgba(34,197,94,0.12)'
      ctx.fill()
      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 3])
      ctx.stroke()
      ctx.setLineDash([])
      // Points
      pts.forEach(([x, y], i) => {
        ctx.beginPath()
        ctx.arc(x, y, 5, 0, Math.PI * 2)
        ctx.fillStyle = '#22c55e'
        ctx.fill()
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(`${i}`, x, y - 9)
      })
    }

    // Tripwires
    tripwires.forEach((tw, ti) => {
      const [x1, y1] = toCanvas(tw.p1)
      const [x2, y2] = toCanvas(tw.p2)
      // Line
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2.5
      ctx.stroke()
      // Direction arrow at midpoint
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2
      const angle = Math.atan2(y2 - y1, x2 - x1)
      const perpAngle = tw.direction === 'left_to_right' ? angle - Math.PI / 2 : angle + Math.PI / 2
      const arrowLen = 14
      ctx.beginPath()
      ctx.moveTo(mx, my)
      ctx.lineTo(mx + Math.cos(perpAngle) * arrowLen, my + Math.sin(perpAngle) * arrowLen)
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.stroke()
      // Endpoints
      ;[[x1, y1], [x2, y2]].forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x, y, 5, 0, Math.PI * 2)
        ctx.fillStyle = '#f59e0b'
        ctx.fill()
      })
      // Label
      ctx.fillStyle = '#fbbf24'
      ctx.font = 'bold 11px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(tw.name || `TW${ti}`, mx, my - 10)
    })

    // Draft tripwire line
    if (twDraft && mousePos) {
      const [x1, y1] = toCanvas(twDraft)
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(mousePos[0], mousePos[1])
      ctx.strokeStyle = 'rgba(245,158,11,0.5)'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.stroke()
      ctx.setLineDash([])
    }
  }, [roi, tripwires, scale, width, height, toCanvas, twDraft, mousePos, imgVer])

  useEffect(() => { draw() }, [draw])

  // ---- Hit test ----
  const hitTest = useCallback((cx: number, cy: number) => {
    const r = 10
    // ROI points
    for (let i = 0; i < roi.length; i++) {
      const [px, py] = toCanvas(roi[i])
      if (Math.hypot(cx - px, cy - py) < r) return { type: 'roi' as const, idx: i }
    }
    // Tripwire endpoints
    for (let ti = 0; ti < tripwires.length; ti++) {
      for (const ptKey of ['p1', 'p2'] as const) {
        const [px, py] = toCanvas(tripwires[ti][ptKey])
        if (Math.hypot(cx - px, cy - py) < r) return { type: 'tw' as const, twIdx: ti, ptKey, idx: ti }
      }
    }
    return null
  }, [roi, tripwires, toCanvas])

  // ---- Mouse handlers ----
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const [cx, cy] = getCanvasXY(e)
    const real = toReal(cx, cy)

    if (mode === 'select') {
      const hit = hitTest(cx, cy)
      if (hit) setDragIdx(hit)
      return
    }

    if (mode === 'roi') {
      onRoiChange([...roi, real])
      return
    }

    if (mode === 'tripwire') {
      if (!twDraft) {
        setTwDraft(real)
      } else {
        const newTw: Tripwire = {
          id: `tw${String(Date.now()).slice(-4)}`,
          name: `线${tripwires.length + 1}`,
          p1: twDraft,
          p2: real,
          direction: 'left_to_right',
          cooldown: 0.5,
        }
        onTripwiresChange([...tripwires, newTw])
        setTwDraft(null)
        setMousePos(null)
      }
    }
  }, [mode, roi, tripwires, twDraft, hitTest, getCanvasXY, toReal, onRoiChange, onTripwiresChange])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const [cx, cy] = getCanvasXY(e)

    if (mode === 'tripwire' && twDraft) {
      setMousePos([cx, cy])
    }

    if (dragIdx) {
      const real = toReal(cx, cy)
      if (dragIdx.type === 'roi') {
        const next = [...roi]
        next[dragIdx.idx] = real
        onRoiChange(next)
      } else if (dragIdx.type === 'tw' && dragIdx.twIdx !== undefined && dragIdx.ptKey) {
        const next = tripwires.map((tw, i) =>
          i === dragIdx.twIdx ? { ...tw, [dragIdx.ptKey!]: real } : tw
        )
        onTripwiresChange(next)
      }
    }
  }, [mode, dragIdx, twDraft, roi, tripwires, getCanvasXY, toReal, onRoiChange, onTripwiresChange])

  const handleMouseUp = useCallback(() => { setDragIdx(null) }, [])

  const handleRightClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    if (mode === 'roi' && roi.length > 0) {
      onRoiChange(roi.slice(0, -1))
    }
    if (mode === 'tripwire' && twDraft) {
      setTwDraft(null)
      setMousePos(null)
    }
  }, [mode, roi, twDraft, onRoiChange])

  // ---- Tripwire list helpers ----
  const removeTw = (idx: number) => onTripwiresChange(tripwires.filter((_, i) => i !== idx))
  const updateTw = (idx: number, patch: Partial<Tripwire>) =>
    onTripwiresChange(tripwires.map((tw, i) => i === idx ? { ...tw, ...patch } : tw))

  const canvasH = height * scale

  return (
    <div className="flex flex-col gap-3">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 text-xs">
        <ToolBtn active={mode === 'select'} onClick={() => { setMode('select'); setTwDraft(null) }}>
          <MousePointer size={13} /> 选择
        </ToolBtn>
        <ToolBtn active={mode === 'roi'} onClick={() => { setMode('roi'); setTwDraft(null) }}>
          <Plus size={13} /> 画区域
        </ToolBtn>
        <ToolBtn active={mode === 'tripwire'} onClick={() => { setMode('tripwire') }}>
          <Minus size={13} /> 画入口线
        </ToolBtn>
        <span className="mx-1 w-px h-4 bg-border" />
        <button onClick={() => onRoiChange([])} className="px-2 py-1 rounded-lg text-muted hover:text-danger hover:bg-danger-dim cursor-pointer transition-colors flex items-center gap-1">
          <Trash2 size={11} /> 清除区域
        </button>
        <button onClick={loadSnapshot} className="px-2 py-1 rounded-lg text-muted hover:text-foreground hover:bg-card cursor-pointer transition-colors flex items-center gap-1">
          <RefreshCw size={11} /> 刷新画面
        </button>
        <span className="flex-1" />
        <span className="text-muted/50">
          {mode === 'roi' ? '左键添加点 · 右键撤销' : mode === 'tripwire' ? '点击两点画线 · 右键取消' : '拖拽移动点位'}
        </span>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="relative rounded-xl overflow-hidden border border-border bg-bg-elevated" style={{ height: canvasH || 300 }}>
        <canvas
          ref={canvasRef}
          className="absolute inset-0 cursor-crosshair"
          style={{ width: '100%', height: '100%' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onContextMenu={handleRightClick}
        />
      </div>

      {/* Tripwire list */}
      {tripwires.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[11px] text-muted">入口线列表</span>
          {tripwires.map((tw, i) => (
            <div key={tw.id} className="flex items-center gap-2 bg-bg-elevated rounded-xl px-3 py-2 text-sm border border-border">
              <span className="w-2 h-2 rounded-full bg-warning flex-shrink-0" />
              <input
                value={tw.name}
                onChange={e => updateTw(i, { name: e.target.value })}
                className="bg-transparent flex-1 min-w-0 focus:outline-none text-sm"
                placeholder="线名称"
              />
              <select
                value={tw.direction}
                onChange={e => updateTw(i, { direction: e.target.value })}
                className="bg-card border border-border rounded-lg px-2 py-1 text-xs cursor-pointer"
              >
                <option value="left_to_right">左→右 = 进入</option>
                <option value="right_to_left">右→左 = 进入</option>
              </select>
              <button onClick={() => removeTw(i)} className="p-1 rounded-lg hover:bg-danger-dim text-muted hover:text-danger cursor-pointer transition-colors">
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ToolBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg cursor-pointer transition-colors ${
        active ? 'bg-accent text-white' : 'text-muted hover:text-foreground hover:bg-card'
      }`}
    >
      {children}
    </button>
  )
}

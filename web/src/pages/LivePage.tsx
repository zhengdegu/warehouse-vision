import { useEffect, useState, useRef, useCallback } from 'react'
import { api } from '../api'
import { AlertPanel } from '../components/AlertPanel'
import { ChevronDown, ArrowDownLeft, ArrowUpRight, AlertTriangle, Radio, Grid2x2, Maximize2, Camera, Users, Car } from 'lucide-react'

type CameraInfo = { id: string; name: string }
type AreaCounts = { total?: number; person?: number; car?: number; truck?: number; bus?: number; motorcycle?: number; bicycle?: number }
type Counts = Record<string, number | AreaCounts>

export function LivePage() {
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [current, setCurrent] = useState('')
  const [counts, setCounts] = useState<Counts>({})
  const [splitView, setSplitView] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    api.getCameras().then(cams => {
      setCameras(cams)
      if (cams.length > 0 && !current) setCurrent(cams[0].id)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (splitView || !current || !imgRef.current) return
    const img = imgRef.current
    img.src = ''
    const t = setTimeout(() => { img.src = `/stream/${current}?t=${Date.now()}` }, 50)
    return () => { clearTimeout(t); img.src = '' }
  }, [current, splitView])

  useEffect(() => {
    if (!current) return
    const poll = () => api.getCounts().then(d => setCounts(d[current] || {})).catch(() => {})
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [current])

  const takeScreenshot = useCallback(async () => {
    if (!current) return
    try {
      const res = await fetch(`/snapshot/${current}`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${current}_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`
      a.click()
      URL.revokeObjectURL(url)
    } catch {}
  }, [current])

  const area = (counts.area || {}) as AreaCounts
  const stats = [
    { label: '今日进入', val: (counts.today_in as number) ?? 0, icon: ArrowDownLeft, cls: 'text-success', bg: 'bg-success-dim' },
    { label: '今日离开', val: (counts.today_out as number) ?? 0, icon: ArrowUpRight, cls: 'text-warning', bg: 'bg-warning-dim' },
    { label: '区域人数', val: area.person ?? 0, icon: Users, cls: 'text-info', bg: 'bg-info-dim' },
    { label: '区域车辆', val: (area.car ?? 0) + (area.truck ?? 0) + (area.bus ?? 0), icon: Car, cls: 'text-accent', bg: 'bg-accent-dim' },
    { label: '告警', val: (counts.alert_count as number) ?? 0, icon: AlertTriangle, cls: 'text-danger', bg: 'bg-danger-dim' },
  ]

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col p-4 gap-3 min-w-0">
        <div className="flex items-center gap-3">
          {!splitView && (
            <div className="relative">
              <select
                value={current}
                onChange={e => setCurrent(e.target.value)}
                className="appearance-none bg-card border border-border rounded-xl px-4 py-2 pr-9 text-sm cursor-pointer transition-colors hover:border-border-light focus:outline-none focus:border-accent"
              >
                {cameras.map(c => (
                  <option key={c.id} value={c.id}>{c.name || c.id}</option>
                ))}
              </select>
              <ChevronDown size={13} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
            </div>
          )}
          <span className="flex items-center gap-1.5 text-[11px] text-success/70">
            <Radio size={12} className="animate-pulse" />
            LIVE
          </span>
          <div className="ml-auto flex items-center gap-2">
            {!splitView && (
              <button onClick={takeScreenshot} className="flex items-center gap-1.5 px-3 py-1.5 bg-card border border-border rounded-xl text-xs text-muted hover:text-foreground hover:border-border-light cursor-pointer transition-colors" title="截图">
                <Camera size={13} /> 截图
              </button>
            )}
            <button
              onClick={() => setSplitView(!splitView)}
              className={`flex items-center gap-1.5 px-3 py-1.5 border rounded-xl text-xs cursor-pointer transition-colors ${splitView ? 'bg-accent-dim border-accent text-accent' : 'bg-card border-border text-muted hover:text-foreground hover:border-border-light'}`}
              title={splitView ? '单画面' : '分屏'}
            >
              {splitView ? <Maximize2 size={13} /> : <Grid2x2 size={13} />}
              {splitView ? '单画面' : '分屏'}
            </button>
          </div>
        </div>

        {splitView ? (
          <SplitView cameras={cameras} onSelect={id => { setSplitView(false); setCurrent(id) }} />
        ) : (
          <>
            <div className="flex-1 bg-bg-elevated rounded-2xl overflow-hidden relative min-h-0 border border-border">
              <img ref={imgRef} alt="视频流" className="w-full h-full object-contain" />
              {!current && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-muted/40 gap-2">
                  <Radio size={36} />
                  <span className="text-sm">无摄像头信号</span>
                </div>
              )}
              <div className="absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-bg/60 to-transparent pointer-events-none" />
            </div>

            <div className="grid grid-cols-5 gap-3">
              {stats.map(({ label, val, icon: Icon, cls, bg }) => (
                <div key={label} className="bg-card rounded-xl p-3 border border-border hover:border-border-light transition-colors cursor-default">
                  <div className="flex items-center justify-between">
                    <span className={`w-7 h-7 rounded-lg ${bg} flex items-center justify-center`}>
                      <Icon size={14} className={cls} />
                    </span>
                    <span className={`text-xl font-bold font-mono tabular-nums ${cls}`}>{val}</span>
                  </div>
                  <span className="text-[11px] text-muted mt-1 block">{label}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <AlertPanel cameraId={current} />
    </div>
  )
}

function SplitView({ cameras, onSelect }: { cameras: CameraInfo[]; onSelect: (id: string) => void }) {
  const cols = cameras.length <= 2 ? 1 : cameras.length <= 4 ? 2 : 3
  return (
    <div className={`flex-1 grid gap-2 min-h-0 overflow-y-auto`} style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {cameras.map(cam => (
        <SplitCell key={cam.id} cam={cam} onSelect={onSelect} />
      ))}
    </div>
  )
}

function SplitCell({ cam, onSelect }: { cam: CameraInfo; onSelect: (id: string) => void }) {
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    if (!imgRef.current) return
    const img = imgRef.current
    img.src = ''
    const t = setTimeout(() => { img.src = `/stream/${cam.id}?t=${Date.now()}` }, 50)
    return () => { clearTimeout(t); img.src = '' }
  }, [cam.id])

  return (
    <div
      onClick={() => onSelect(cam.id)}
      className="bg-bg-elevated rounded-xl overflow-hidden relative border border-border hover:border-accent cursor-pointer transition-colors group"
    >
      <img ref={imgRef} alt={cam.name} className="w-full h-full object-contain" />
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent p-2">
        <span className="text-xs text-white/80 font-medium">{cam.name || cam.id}</span>
      </div>
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <Maximize2 size={14} className="text-white/70" />
      </div>
    </div>
  )
}

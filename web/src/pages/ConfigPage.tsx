import { useEffect, useState } from 'react'
import { api } from '../api'
import { Plus, Pencil, Trash2, Camera, Wifi, WifiOff } from 'lucide-react'
import { CameraModal } from '../components/CameraModal'

type Cam = { id: string; name: string }
type Detail = Record<string, unknown>

export function ConfigPage() {
  const [cameras, setCameras] = useState<Cam[]>([])
  const [details, setDetails] = useState<Record<string, Detail>>({})
  const [health, setHealth] = useState<Record<string, { running: boolean }>>({})
  const [modal, setModal] = useState<{ mode: 'add' | 'edit'; camId?: string } | null>(null)

  const load = async () => {
    const cams = await api.getCameras()
    setCameras(cams)
    const d: Record<string, Detail> = {}
    for (const c of cams) { try { d[c.id] = await api.getCamera(c.id) } catch {} }
    setDetails(d)
    try {
      const h = await api.getSystemHealth()
      setHealth((h.cameras || {}) as Record<string, { running: boolean }>)
    } catch {}
  }

  useEffect(() => { load() }, [])

  const del = async (id: string) => {
    if (!confirm(`确定删除摄像头 ${id}？`)) return
    try { await api.deleteCamera(id); load() } catch {}
  }

  return (
    <div className="p-5 h-full flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-base font-semibold">摄像头配置</h1>
        <button
          onClick={() => setModal({ mode: 'add' })}
          className="flex items-center gap-1.5 px-4 py-2 bg-accent text-white rounded-xl text-sm font-medium cursor-pointer hover:bg-accent-light transition-colors"
        >
          <Plus size={14} /> 添加
        </button>
      </div>

      {cameras.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-muted gap-3">
          <Camera size={32} className="opacity-20" />
          <span className="text-sm">暂无摄像头</span>
          <button onClick={() => setModal({ mode: 'add' })} className="text-accent text-sm cursor-pointer hover:underline">添加第一个</button>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 overflow-y-auto content-start">
          {cameras.map(cam => {
            const d = details[cam.id] || {}
            return (
              <div key={cam.id} className="bg-card rounded-2xl border border-border p-5 hover:border-border-light transition-all group">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="w-9 h-9 rounded-xl bg-accent-dim flex items-center justify-center">
                      <Camera size={16} className="text-accent" />
                    </span>
                    <div>
                      <div className="text-sm font-medium">{cam.name || cam.id}</div>
                      <div className="text-[11px] text-muted font-mono">{cam.id}</div>
                    </div>
                  </div>
                  <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={() => setModal({ mode: 'edit', camId: cam.id })} className="p-2 rounded-lg hover:bg-card-hover text-muted hover:text-foreground cursor-pointer transition-colors">
                      <Pencil size={13} />
                    </button>
                    <button onClick={() => del(cam.id)} className="p-2 rounded-lg hover:bg-danger-dim text-muted hover:text-danger cursor-pointer transition-colors">
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <Chip label="分辨率" value={`${d.width}×${d.height}`} />
                  <Chip label="帧率" value={`${d.fps} fps`} />
                  <Chip label="状态" value={
                    health[cam.id]?.running
                      ? <span className="flex items-center gap-1 text-success"><Wifi size={10} />在线</span>
                      : <span className="flex items-center gap-1 text-danger"><WifiOff size={10} />离线</span>
                  } />
                </div>
                <div className="mt-3 text-[10px] text-muted/50 truncate font-mono">{(d.url as string) || ''}</div>
              </div>
            )
          })}
        </div>
      )}

      {modal && <CameraModal mode={modal.mode} camId={modal.camId} onClose={() => setModal(null)} onSaved={() => { setModal(null); load() }} />}
    </div>
  )
}

function Chip({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-bg-elevated rounded-lg px-2.5 py-2">
      <div className="text-muted mb-0.5">{label}</div>
      <div className="font-medium text-foreground">{value}</div>
    </div>
  )
}

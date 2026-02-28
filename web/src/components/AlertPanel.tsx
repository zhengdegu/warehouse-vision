import { useEffect, useState, useCallback } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { api } from '../api'
import { Clock, History, X, AlertTriangle, ShieldAlert, Footprints, Eye } from 'lucide-react'

type Evt = Record<string, unknown>

const META: Record<string, { label: string; icon: typeof AlertTriangle; cls: string }> = {
  intrusion: { label: '区域入侵', icon: ShieldAlert, cls: 'text-danger' },
  tripwire:  { label: '越线检测', icon: Footprints, cls: 'text-warning' },
  presence:  { label: '目标检测', icon: Eye, cls: 'text-info' },
  anomaly:   { label: '异常检测', icon: AlertTriangle, cls: 'text-danger' },
}
const SUB: Record<string, string> = { dwell:'滞留', crowd:'聚集', proximity:'人车过近', fight:'打架', fall:'跌倒' }

function meta(e: Evt) {
  const m = META[e.type as string] || { label: e.type as string, icon: AlertTriangle, cls: 'text-muted' }
  const sub = e.sub_type ? ` · ${SUB[e.sub_type as string] || e.sub_type}` : ''
  return { ...m, label: m.label + sub }
}

export function AlertPanel({ cameraId }: { cameraId: string }) {
  const [alerts, setAlerts] = useState<Evt[]>([])
  const [detail, setDetail] = useState<Evt | null>(null)
  const { subscribe } = useWebSocket()

  useEffect(() => subscribe((e) => {
    if (e.camera_id && e.camera_id !== cameraId) return
    setAlerts(p => [e, ...p].slice(0, 100))
  }), [subscribe, cameraId])

  useEffect(() => { setAlerts([]) }, [cameraId])

  const loadHistory = useCallback(async () => {
    if (!cameraId) return
    try { const d = await api.getEvents(`limit=50&camera_id=${cameraId}`); setAlerts(d.items || []) } catch {}
  }, [cameraId])

  return (
    <>
      <aside className="w-80 bg-surface border-l border-border flex flex-col shrink-0">
        {/* Header */}
        <div className="flex items-center justify-between px-4 h-12 border-b border-border shrink-0">
          <span className="flex items-center gap-2 text-sm font-medium">
            <AlertTriangle size={13} className="text-danger" />
            实时告警
            {alerts.length > 0 && (
              <span className="min-w-[18px] h-[18px] px-1 rounded-full text-[10px] font-semibold bg-danger-dim text-danger flex items-center justify-center">
                {alerts.length}
              </span>
            )}
          </span>
          <button onClick={loadHistory} className="text-[11px] text-muted hover:text-muted-light flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-card cursor-pointer transition-colors">
            <History size={11} /> 历史
          </button>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted gap-2 opacity-40">
              <ShieldAlert size={28} />
              <span className="text-xs">暂无告警</span>
            </div>
          ) : alerts.map((evt, i) => {
            const m = meta(evt)
            const Icon = m.icon
            return (
              <button
                key={i}
                onClick={() => setDetail(evt)}
                className="w-full text-left px-4 py-3 border-b border-border/40 hover:bg-card cursor-pointer transition-colors animate-fade-up"
                style={{ animationDelay: `${Math.min(i * 30, 200)}ms` }}
              >
                <div className="flex items-center gap-2">
                  <Icon size={12} className={m.cls} />
                  <span className={`text-xs font-medium ${m.cls}`}>{m.label}</span>
                  <span className="ml-auto text-[10px] text-muted font-mono flex items-center gap-1">
                    <Clock size={9} />
                    {new Date((evt.timestamp as number) * 1000).toLocaleTimeString('zh-CN')}
                  </span>
                </div>
                <p className="text-[11px] text-muted mt-1 truncate pl-5">
                  {(evt.detail as string) || `cam: ${evt.camera_id}`}
                </p>
              </button>
            )
          })}
        </div>
      </aside>

      {/* Detail */}
      {detail && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6" onClick={() => setDetail(null)}>
          <div className="bg-surface rounded-2xl max-w-[680px] w-full max-h-[85vh] overflow-y-auto p-6 border border-border animate-fade-up" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <span className={`text-sm font-semibold ${meta(detail).cls}`}>{meta(detail).label}</span>
              <button onClick={() => setDetail(null)} className="w-7 h-7 rounded-lg flex items-center justify-center text-muted hover:text-foreground hover:bg-card cursor-pointer transition-colors">
                <X size={15} />
              </button>
            </div>
            {detail.screenshot ? (
              <img src={`/events/${encodeURIComponent(detail.screenshot as string)}`} alt="告警截图" className="w-full rounded-xl mb-4 bg-black border border-border" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
            ) : (
              <div className="w-full h-44 bg-bg-elevated rounded-xl mb-4 flex items-center justify-center text-muted text-sm border border-border">无截图</div>
            )}
            <dl className="grid grid-cols-[90px_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-muted">时间</dt>
              <dd className="font-mono text-xs">{new Date((detail.timestamp as number) * 1000).toLocaleString('zh-CN')}</dd>
              <dt className="text-muted">摄像头</dt>
              <dd>{(detail.camera_id as string) || '-'}</dd>
              {detail.track_id !== undefined && <><dt className="text-muted">跟踪ID</dt><dd className="font-mono text-accent">#{String(detail.track_id)}</dd></>}
              {detail.class_name ? <><dt className="text-muted">类别</dt><dd>{String(detail.class_name)}</dd></> : null}
              {detail.detail ? <><dt className="text-muted">详情</dt><dd>{String(detail.detail)}</dd></> : null}
            </dl>
          </div>
        </div>
      )}
    </>
  )
}

import { useEffect, useState } from 'react'
import { api } from '../api'
import { ChevronLeft, ChevronRight, Image, X, ShieldAlert, Filter } from 'lucide-react'

type EventItem = Record<string, unknown>

const TYPE_BADGE: Record<string, { label: string; cls: string }> = {
  intrusion: { label: '入侵', cls: 'bg-danger-dim text-danger' },
  tripwire:  { label: '越线', cls: 'bg-warning-dim text-warning' },
  presence:  { label: '检测', cls: 'bg-info-dim text-info' },
  counting:  { label: '计数', cls: 'bg-accent-dim text-accent' },
  anomaly:   { label: '异常', cls: 'bg-danger-dim text-danger' },
}
const SUB: Record<string, string> = { dwell:'滞留', crowd:'聚集', proximity:'过近', fight:'打架', fall:'跌倒' }
const TYPE_OPTIONS = [
  { value: '', label: '全部类型' },
  { value: 'intrusion', label: '入侵' },
  { value: 'tripwire', label: '越线' },
  { value: 'presence', label: '检测' },
  { value: 'anomaly', label: '异常' },
]

export function EventsPage() {
  const [events, setEvents] = useState<EventItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [camFilter, setCamFilter] = useState('')
  const [typeFilter, setTypeFilter] = useState('')
  const [cameras, setCameras] = useState<{ id: string; name: string; timezone?: string }[]>([])
  const [detail, setDetail] = useState<EventItem | null>(null)
  const [summary, setSummary] = useState<Record<string, unknown>>({})
  const limit = 30

  useEffect(() => { api.getCameras().then(setCameras).catch(() => {}) }, [])

  // 摄像头时区映射
  const tzMap: Record<string, string> = {}
  for (const c of cameras) {
    if (c.timezone) tzMap[c.id] = c.timezone
  }
  const fmtTime = (ts: number, cameraId?: string) => {
    const tz = cameraId ? tzMap[cameraId] : undefined
    return new Date(ts * 1000).toLocaleString('zh-CN', tz ? { timeZone: tz } : undefined)
  }
  useEffect(() => { api.getEventSummary().then(setSummary).catch(() => {}) }, [])

  useEffect(() => {
    const p = new URLSearchParams({ limit: String(limit), offset: String(page * limit) })
    if (camFilter) p.set('camera_id', camFilter)
    if (typeFilter) p.set('event_type', typeFilter)
    api.getEvents(p.toString()).then(d => { setEvents(d.items || []); setTotal(d.total || 0) }).catch(() => {})
  }, [page, camFilter, typeFilter])

  const badge = (e: EventItem) => {
    const b = TYPE_BADGE[e.type as string] || { label: e.type as string, cls: 'bg-card text-muted' }
    const sub = e.sub_type ? `/${SUB[e.sub_type as string] || e.sub_type}` : ''
    return { ...b, label: b.label + sub }
  }

  const summaryByType = (summary.by_type || {}) as Record<string, number>
  const totalEvents = (summary.total || 0) as number

  return (
    <div className="p-5 h-full flex flex-col gap-4">
      {/* Summary cards */}
      <div className="grid grid-cols-5 gap-3">
        <SummaryCard label="总事件" value={totalEvents} cls="text-foreground" />
        <SummaryCard label="入侵" value={summaryByType.intrusion || 0} cls="text-danger" />
        <SummaryCard label="越线" value={summaryByType.tripwire || 0} cls="text-warning" />
        <SummaryCard label="异常" value={summaryByType.anomaly || 0} cls="text-danger" />
        <SummaryCard label="检测" value={summaryByType.presence || 0} cls="text-info" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold">事件记录</h1>
          <span className="text-[11px] text-muted font-mono bg-card px-2 py-0.5 rounded-lg">{total}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Filter size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <select value={typeFilter} onChange={e => { setTypeFilter(e.target.value); setPage(0) }}
              className="appearance-none bg-card border border-border rounded-xl pl-8 pr-8 py-2 text-sm cursor-pointer focus:outline-none focus:border-accent">
              {TYPE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>
          <select value={camFilter} onChange={e => { setCamFilter(e.target.value); setPage(0) }}
            className="appearance-none bg-card border border-border rounded-xl px-3 py-2 text-sm cursor-pointer focus:outline-none focus:border-accent">
            <option value="">全部摄像头</option>
            {cameras.map(c => <option key={c.id} value={c.id}>{c.name || c.id}</option>)}
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto rounded-2xl border border-border bg-surface">
        <table className="w-full text-sm">
          <thead className="bg-card/40 sticky top-0 z-10">
            <tr className="text-muted text-left text-[11px] uppercase tracking-wider">
              <th className="px-4 py-3 font-medium">时间</th>
              <th className="px-4 py-3 font-medium">类型</th>
              <th className="px-4 py-3 font-medium">摄像头</th>
              <th className="px-4 py-3 font-medium">详情</th>
              <th className="px-4 py-3 font-medium w-8"></th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr><td colSpan={5} className="text-center py-20"><ShieldAlert size={26} className="mx-auto mb-2 text-muted/20" /><span className="text-xs text-muted">暂无事件</span></td></tr>
            ) : events.map((evt, i) => {
              const b = badge(evt)
              return (
                <tr key={i} onClick={() => setDetail(evt)} className="border-t border-border/30 hover:bg-card/40 cursor-pointer transition-colors">
                  <td className="px-4 py-3 text-muted whitespace-nowrap font-mono text-xs">{fmtTime(evt.timestamp as number, evt.camera_id as string)}</td>
                  <td className="px-4 py-3"><span className={`px-2 py-0.5 rounded-lg text-[11px] font-medium ${b.cls}`}>{b.label}</span></td>
                  <td className="px-4 py-3 text-muted text-xs">{evt.camera_id as string}</td>
                  <td className="px-4 py-3 text-muted truncate max-w-[280px] text-xs">{(evt.detail as string) || (evt.class_name as string) || '-'}</td>
                  <td className="px-4 py-3">{evt.screenshot ? <Image size={13} className="text-muted-light" /> : null}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-xs text-muted">
        <span className="font-mono">{page * limit + 1}–{Math.min((page + 1) * limit, total)} / {total}</span>
        <div className="flex items-center gap-1">
          <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0} className="p-1.5 rounded-lg hover:bg-card disabled:opacity-20 cursor-pointer transition-colors"><ChevronLeft size={15} /></button>
          <span className="px-2.5 py-1 bg-card rounded-lg font-mono font-medium text-foreground">{page + 1}</span>
          <button onClick={() => setPage(page + 1)} disabled={(page + 1) * limit >= total} className="p-1.5 rounded-lg hover:bg-card disabled:opacity-20 cursor-pointer transition-colors"><ChevronRight size={15} /></button>
        </div>
      </div>

      {/* Detail modal */}
      {detail && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6" onClick={() => setDetail(null)}>
          <div className="bg-surface rounded-2xl max-w-[680px] w-full max-h-[85vh] overflow-y-auto p-6 border border-border animate-fade-up" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <span className={`px-2.5 py-1 rounded-lg text-xs font-medium ${badge(detail).cls}`}>{badge(detail).label}</span>
              <button onClick={() => setDetail(null)} className="w-7 h-7 rounded-lg flex items-center justify-center text-muted hover:text-foreground hover:bg-card cursor-pointer transition-colors"><X size={15} /></button>
            </div>
            {detail.screenshot ? (
              <img src={`/events/${encodeURIComponent(detail.screenshot as string)}`} alt="" className="w-full rounded-xl mb-4 bg-black border border-border" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
            ) : (
              <div className="w-full h-44 bg-bg-elevated rounded-xl mb-4 flex items-center justify-center text-muted text-sm border border-border">无截图</div>
            )}
            <dl className="grid grid-cols-[90px_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-muted">时间</dt><dd className="font-mono text-xs">{fmtTime(detail.timestamp as number, detail.camera_id as string)}</dd>
              <dt className="text-muted">摄像头</dt><dd>{(detail.camera_id as string) || '-'}</dd>
              {detail.class_name ? <><dt className="text-muted">类别</dt><dd>{detail.class_name as string}</dd></> : null}
              {detail.crossing_direction ? <><dt className="text-muted">方向</dt><dd>{detail.crossing_direction === 'in' ? '进入' : '离开'}</dd></> : null}
              {detail.detail ? <><dt className="text-muted">详情</dt><dd>{detail.detail as string}</dd></> : null}
            </dl>
          </div>
        </div>
      )}
    </div>
  )
}

function SummaryCard({ label, value, cls }: { label: string; value: number; cls: string }) {
  return (
    <div className="bg-card rounded-xl border border-border p-3 hover:border-border-light transition-colors">
      <div className={`text-xl font-bold font-mono tabular-nums ${cls}`}>{value}</div>
      <div className="text-[11px] text-muted mt-0.5">{label}</div>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { api } from '../api'
import { BarChart3, Filter, TrendingUp, CheckCircle, AlertTriangle, Plus, X } from 'lucide-react'

type StatItem = Record<string, unknown>
type TrendItem = { timestamp: number; type: string; count: number }
type RecordItem = Record<string, unknown>

const RULE_LABELS: Record<string, string> = {
  intrusion: '入侵检测',
  tripwire: '越线检测',
  presence: '目标检测',
  'anomaly/dwell': '徘徊检测',
  'anomaly/crowd': '聚集检测',
  'anomaly/proximity': '人车过近',
  'anomaly/fight': '打架检测',
  'anomaly/fall': '跌倒检测',
  counting: '流量计数',
}

const RULE_COLORS: Record<string, string> = {
  intrusion: 'bg-danger-dim text-danger',
  tripwire: 'bg-warning-dim text-warning',
  presence: 'bg-info-dim text-info',
  'anomaly/dwell': 'bg-accent-dim text-accent',
  'anomaly/crowd': 'bg-danger-dim text-danger',
  'anomaly/proximity': 'bg-warning-dim text-warning',
  'anomaly/fight': 'bg-danger-dim text-danger',
  'anomaly/fall': 'bg-warning-dim text-warning',
  counting: 'bg-info-dim text-info',
}

export function AccuracyPage() {
  const [stats, setStats] = useState<StatItem[]>([])
  const [summary, setSummary] = useState<Record<string, { total: number; avg_confidence: number; cameras: number }>>({})
  const [trend, setTrend] = useState<TrendItem[]>([])
  const [records, setRecords] = useState<RecordItem[]>([])
  const [cameras, setCameras] = useState<{ id: string; name: string }[]>([])
  const [camFilter, setCamFilter] = useState('')
  const [hours, setHours] = useState(24)
  const [showForm, setShowForm] = useState(false)

  useEffect(() => { api.getCameras().then(setCameras).catch(() => {}) }, [])

  useEffect(() => {
    const p = new URLSearchParams({ hours: String(hours) })
    if (camFilter) p.set('camera_id', camFilter)
    const qs = p.toString()
    api.getAccuracyStats(qs).then(d => { setStats(d.items || []); setSummary((d.summary || {}) as typeof summary) }).catch(() => {})
    api.getAccuracyTrend(qs).then(d => setTrend((d.items || []) as TrendItem[])).catch(() => {})
    api.getAccuracyRecords(camFilter ? `camera_id=${camFilter}` : '').then(d => setRecords(d.items || [])).catch(() => {})
  }, [camFilter, hours])

  // 计算趋势中的最大值用于柱状图
  const trendByType: Record<string, { ts: number; count: number }[]> = {}
  for (const t of trend) {
    if (!trendByType[t.type]) trendByType[t.type] = []
    trendByType[t.type].push({ ts: t.timestamp, count: t.count })
  }

  const totalAlerts = Object.values(summary).reduce((s, v) => s + v.total, 0)
  const avgConf = Object.values(summary).length > 0
    ? Object.values(summary).reduce((s, v) => s + v.avg_confidence * v.total, 0) / (totalAlerts || 1)
    : 0

  return (
    <div className="p-5 h-full flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold">报警准确率统计</h1>
          <span className="text-[11px] text-muted font-mono bg-card px-2 py-0.5 rounded-lg">
            最近 {hours}h
          </span>
        </div>
        <div className="flex items-center gap-2">
          <select value={hours} onChange={e => setHours(Number(e.target.value))}
            className="appearance-none bg-card border border-border rounded-xl px-3 py-2 text-sm cursor-pointer focus:outline-none focus:border-accent">
            <option value={6}>6小时</option>
            <option value={24}>24小时</option>
            <option value={72}>3天</option>
            <option value={168}>7天</option>
          </select>
          <div className="relative">
            <Filter size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <select value={camFilter} onChange={e => setCamFilter(e.target.value)}
              className="appearance-none bg-card border border-border rounded-xl pl-8 pr-8 py-2 text-sm cursor-pointer focus:outline-none focus:border-accent">
              <option value="">全部摄像头</option>
              {cameras.map(c => <option key={c.id} value={c.id}>{c.name || c.id}</option>)}
            </select>
          </div>
          <button onClick={() => setShowForm(true)}
            className="flex items-center gap-1.5 px-3 py-2 bg-accent text-white rounded-xl text-sm font-medium cursor-pointer hover:bg-accent-light transition-colors">
            <Plus size={14} /> 标注
          </button>
        </div>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard icon={<BarChart3 size={16} />} label="总报警数" value={totalAlerts} cls="text-foreground" />
        <StatCard icon={<TrendingUp size={16} />} label="平均置信度" value={`${(avgConf * 100).toFixed(1)}%`} cls="text-accent" />
        <StatCard icon={<CheckCircle size={16} />} label="规则类型" value={Object.keys(summary).length} cls="text-success" />
        <StatCard icon={<AlertTriangle size={16} />} label="活跃摄像头" value={new Set(stats.map(s => s.camera_id)).size} cls="text-warning" />
      </div>

      {/* Summary by rule type */}
      <div className="bg-surface rounded-2xl border border-border p-4">
        <h2 className="text-sm font-medium mb-3 text-muted-light">各规则报警分布</h2>
        {Object.keys(summary).length === 0 ? (
          <div className="text-center py-8 text-muted text-sm">暂无报警数据</div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {Object.entries(summary).sort((a, b) => b[1].total - a[1].total).map(([key, val]) => (
              <div key={key} className="bg-card rounded-xl border border-border p-3 hover:border-border-light transition-colors">
                <div className="flex items-center gap-2 mb-2">
                  <span className={`px-2 py-0.5 rounded-lg text-[10px] font-medium ${RULE_COLORS[key] || 'bg-card text-muted'}`}>
                    {RULE_LABELS[key] || key}
                  </span>
                </div>
                <div className="text-lg font-bold font-mono tabular-nums">{val.total}</div>
                <div className="flex items-center justify-between text-[11px] text-muted mt-1">
                  <span>置信度 {(val.avg_confidence * 100).toFixed(1)}%</span>
                  <span>{val.cameras} 摄像头</span>
                </div>
                {/* Mini bar */}
                <div className="mt-2 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                  <div className="h-full bg-accent rounded-full transition-all" style={{ width: `${Math.min(100, (val.total / (totalAlerts || 1)) * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Detail table + Records side by side */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
        {/* Per-camera detail */}
        <div className="overflow-y-auto rounded-2xl border border-border bg-surface">
          <div className="px-4 py-3 border-b border-border bg-card/40">
            <span className="text-sm font-medium">按摄像头明细</span>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-card/20 sticky top-0">
              <tr className="text-muted text-left text-[11px] uppercase tracking-wider">
                <th className="px-4 py-2 font-medium">摄像头</th>
                <th className="px-4 py-2 font-medium">规则</th>
                <th className="px-4 py-2 font-medium text-right">报警数</th>
                <th className="px-4 py-2 font-medium text-right">置信度</th>
              </tr>
            </thead>
            <tbody>
              {stats.length === 0 ? (
                <tr><td colSpan={4} className="text-center py-10 text-muted text-xs">暂无数据</td></tr>
              ) : stats.map((s, i) => {
                const key = s.sub_type ? `${s.rule_type}/${s.sub_type}` : s.rule_type as string
                return (
                  <tr key={i} className="border-t border-border/30 hover:bg-card/40 transition-colors">
                    <td className="px-4 py-2 text-xs text-muted font-mono">{s.camera_id as string}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded-lg text-[10px] font-medium ${RULE_COLORS[key] || 'bg-card text-muted'}`}>
                        {RULE_LABELS[key] || key}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">{s.total_alerts as number}</td>
                    <td className="px-4 py-2 text-right font-mono text-xs">{((s.avg_confidence as number) * 100).toFixed(1)}%</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Accuracy records */}
        <div className="overflow-y-auto rounded-2xl border border-border bg-surface">
          <div className="px-4 py-3 border-b border-border bg-card/40">
            <span className="text-sm font-medium">评估记录</span>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-card/20 sticky top-0">
              <tr className="text-muted text-left text-[11px] uppercase tracking-wider">
                <th className="px-4 py-2 font-medium">规则</th>
                <th className="px-4 py-2 font-medium text-right">精确率</th>
                <th className="px-4 py-2 font-medium text-right">召回率</th>
                <th className="px-4 py-2 font-medium text-right">F1</th>
                <th className="px-4 py-2 font-medium text-right">TP/FP/FN</th>
              </tr>
            </thead>
            <tbody>
              {records.length === 0 ? (
                <tr><td colSpan={5} className="text-center py-10 text-muted text-xs">暂无评估记录，点击"标注"添加</td></tr>
              ) : records.map((r, i) => {
                const key = r.sub_type ? `${r.rule_type}/${r.sub_type}` : r.rule_type as string
                const p = r.precision as number
                const rec = r.recall as number
                const f1 = r.f1 as number
                return (
                  <tr key={i} className="border-t border-border/30 hover:bg-card/40 transition-colors">
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded-lg text-[10px] font-medium ${RULE_COLORS[key] || 'bg-card text-muted'}`}>
                        {RULE_LABELS[key] || key}
                      </span>
                      {r.camera_id ? <span className="ml-1 text-[10px] text-muted font-mono">{r.camera_id as string}</span> : null}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      <span className={p >= 0.8 ? 'text-success' : p >= 0.5 ? 'text-warning' : 'text-danger'}>{(p * 100).toFixed(1)}%</span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      <span className={rec >= 0.8 ? 'text-success' : rec >= 0.5 ? 'text-warning' : 'text-danger'}>{(rec * 100).toFixed(1)}%</span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      <span className={f1 >= 0.8 ? 'text-success' : f1 >= 0.5 ? 'text-warning' : 'text-danger'}>{(f1 * 100).toFixed(1)}%</span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-[10px] text-muted">
                      {r.confirmed as number}/{r.false_positive as number}/{r.missed as number}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Annotation form modal */}
      {showForm && <AnnotationModal cameras={cameras} onClose={() => setShowForm(false)} onSaved={() => {
        setShowForm(false)
        // Refresh records
        api.getAccuracyRecords(camFilter ? `camera_id=${camFilter}` : '').then(d => setRecords(d.items || [])).catch(() => {})
      }} />}
    </div>
  )
}

function StatCard({ icon, label, value, cls }: { icon: React.ReactNode; label: string; value: number | string; cls: string }) {
  return (
    <div className="bg-card rounded-xl border border-border p-3 hover:border-border-light transition-colors">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-muted">{icon}</span>
        <span className="text-[11px] text-muted">{label}</span>
      </div>
      <div className={`text-xl font-bold font-mono tabular-nums ${cls}`}>{value}</div>
    </div>
  )
}

const RULE_OPTIONS = [
  { value: 'intrusion', label: '入侵检测', sub: '' },
  { value: 'tripwire', label: '越线检测', sub: '' },
  { value: 'presence', label: '目标检测', sub: '' },
  { value: 'counting', label: '流量计数', sub: '' },
  { value: 'anomaly', label: '徘徊检测', sub: 'dwell' },
  { value: 'anomaly', label: '聚集检测', sub: 'crowd' },
  { value: 'anomaly', label: '人车过近', sub: 'proximity' },
  { value: 'anomaly', label: '打架检测', sub: 'fight' },
  { value: 'anomaly', label: '跌倒检测', sub: 'fall' },
]

function AnnotationModal({ cameras, onClose, onSaved }: {
  cameras: { id: string; name: string }[]
  onClose: () => void
  onSaved: () => void
}) {
  const [form, setForm] = useState({
    camera_id: '',
    rule_idx: 0,
    total_alerts: 0,
    confirmed: 0,
    false_positive: 0,
    missed: 0,
    period_start: '',
    period_end: '',
  })
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      const rule = RULE_OPTIONS[form.rule_idx]
      await api.saveAccuracyRecord({
        camera_id: form.camera_id,
        rule_type: rule.value,
        sub_type: rule.sub,
        total_alerts: form.total_alerts,
        confirmed: form.confirmed,
        false_positive: form.false_positive,
        missed: form.missed,
        period_start: form.period_start ? new Date(form.period_start).getTime() / 1000 : 0,
        period_end: form.period_end ? new Date(form.period_end).getTime() / 1000 : 0,
      })
      onSaved()
    } catch (e: unknown) { alert((e as Error).message) }
    finally { setSaving(false) }
  }

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6" onClick={onClose}>
      <div className="bg-surface rounded-2xl max-w-[520px] w-full p-6 border border-border animate-fade-up" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold">标注准确率评估</h3>
          <button onClick={onClose} className="w-7 h-7 rounded-lg flex items-center justify-center text-muted hover:text-foreground hover:bg-card cursor-pointer transition-colors"><X size={15} /></button>
        </div>
        <div className="space-y-3 text-sm">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted mb-1 block">摄像头</label>
              <select value={form.camera_id} onChange={e => setForm({ ...form, camera_id: e.target.value })}
                className="w-full bg-card border border-border rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-accent">
                <option value="">全部</option>
                {cameras.map(c => <option key={c.id} value={c.id}>{c.name || c.id}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">规则类型</label>
              <select value={form.rule_idx} onChange={e => setForm({ ...form, rule_idx: Number(e.target.value) })}
                className="w-full bg-card border border-border rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-accent">
                {RULE_OPTIONS.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-4 gap-3">
            <NumField label="总报警" value={form.total_alerts} onChange={v => setForm({ ...form, total_alerts: v })} />
            <NumField label="确认(TP)" value={form.confirmed} onChange={v => setForm({ ...form, confirmed: v })} />
            <NumField label="误报(FP)" value={form.false_positive} onChange={v => setForm({ ...form, false_positive: v })} />
            <NumField label="漏报(FN)" value={form.missed} onChange={v => setForm({ ...form, missed: v })} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted mb-1 block">评估开始</label>
              <input type="datetime-local" value={form.period_start} onChange={e => setForm({ ...form, period_start: e.target.value })}
                className="w-full bg-card border border-border rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-accent" />
            </div>
            <div>
              <label className="text-xs text-muted mb-1 block">评估结束</label>
              <input type="datetime-local" value={form.period_end} onChange={e => setForm({ ...form, period_end: e.target.value })}
                className="w-full bg-card border border-border rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-accent" />
            </div>
          </div>
          {/* Preview metrics */}
          {form.confirmed + form.false_positive + form.missed > 0 && (
            <div className="bg-bg-elevated rounded-xl p-3 grid grid-cols-3 gap-3 text-center">
              <div>
                <div className="text-[10px] text-muted mb-0.5">精确率</div>
                <div className="font-mono font-bold text-sm">
                  {((form.confirmed / Math.max(1, form.confirmed + form.false_positive)) * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-[10px] text-muted mb-0.5">召回率</div>
                <div className="font-mono font-bold text-sm">
                  {((form.confirmed / Math.max(1, form.confirmed + form.missed)) * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-[10px] text-muted mb-0.5">F1</div>
                <div className="font-mono font-bold text-sm">
                  {(() => {
                    const p = form.confirmed / Math.max(1, form.confirmed + form.false_positive)
                    const r = form.confirmed / Math.max(1, form.confirmed + form.missed)
                    return ((2 * p * r) / Math.max(0.001, p + r) * 100).toFixed(1)
                  })()}%
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="flex justify-end gap-3 mt-5">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded-xl text-sm cursor-pointer hover:bg-card transition-colors">取消</button>
          <button onClick={save} disabled={saving} className="px-5 py-2 bg-accent text-white rounded-xl text-sm font-medium cursor-pointer hover:bg-accent-light transition-colors disabled:opacity-40">
            {saving ? '保存中…' : '保存'}
          </button>
        </div>
      </div>
    </div>
  )
}

function NumField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <label className="text-xs text-muted mb-1 block">{label}</label>
      <input type="number" min={0} value={value} onChange={e => onChange(Number(e.target.value))}
        className="w-full bg-card border border-border rounded-xl px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent" />
    </div>
  )
}

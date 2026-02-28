import { useEffect, useState } from 'react'
import { api } from '../api'
import { Activity, Clock, Camera, BarChart3, Cpu, Box, Settings, Save, Loader2 } from 'lucide-react'

export function SystemPage() {
  const [stats, setStats] = useState<Record<string, unknown>>({})
  const [health, setHealth] = useState<Record<string, unknown>>({})
  const [sysConfig, setSysConfig] = useState<Record<string, unknown>>({})
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null)

  // Editable model fields
  const [modelPath, setModelPath] = useState('')
  const [analyzeFps, setAnalyzeFps] = useState(5)
  const [confidence, setConfidence] = useState(0.5)
  const [classes, setClasses] = useState('')
  const [poseEnabled, setPoseEnabled] = useState(false)
  const [posePath, setPosePath] = useState('')
  const [poseConf, setPoseConf] = useState(0.3)

  // Editable system fields
  const [logLevel, setLogLevel] = useState('INFO')
  const [statsInterval, setStatsInterval] = useState(60)

  // Editable events fields
  const [screenshot, setScreenshot] = useState(true)
  const [drawBbox, setDrawBbox] = useState(true)
  const [drawRoi, setDrawRoi] = useState(true)
  const [drawTripwire, setDrawTripwire] = useState(true)

  useEffect(() => {
    const load = () => {
      api.getSystemStats().then(setStats).catch(() => {})
      api.getSystemHealth().then(setHealth).catch(() => {})
    }
    load()
    const id = setInterval(load, 5000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    api.getSystemConfig().then(cfg => {
      setSysConfig(cfg)
      syncFromConfig(cfg)
    }).catch(() => {})
  }, [])

  function syncFromConfig(cfg: Record<string, unknown>) {
    const m = (cfg.model || {}) as Record<string, unknown>
    const s = (cfg.system || {}) as Record<string, unknown>
    const e = (cfg.events || {}) as Record<string, unknown>
    const p = (m.pose || {}) as Record<string, unknown>
    setModelPath((m.path as string) || '')
    setAnalyzeFps((m.analyze_fps as number) ?? 5)
    setConfidence((m.confidence as number) ?? 0.5)
    setClasses(Array.isArray(m.classes) ? (m.classes as number[]).join(', ') : '')
    setPoseEnabled(!!p.enabled)
    setPosePath((p.path as string) || '')
    setPoseConf((p.confidence as number) ?? 0.3)
    setLogLevel((s.log_level as string) || 'INFO')
    setStatsInterval((s.stats_interval as number) ?? 60)
    setScreenshot(e.screenshot !== false)
    setDrawBbox(e.draw_bbox !== false)
    setDrawRoi(e.draw_roi !== false)
    setDrawTripwire(e.draw_tripwire !== false)
  }

  function showToast(msg: string, ok: boolean) {
    setToast({ msg, ok })
    setTimeout(() => setToast(null), 3000)
  }

  async function handleSave() {
    setSaving(true)
    try {
      const classArr = classes.trim() ? classes.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)) : []
      const payload: Record<string, unknown> = {
        model: {
          path: modelPath,
          analyze_fps: analyzeFps,
          confidence,
          classes: classArr,
          pose: { enabled: poseEnabled, path: posePath, confidence: poseConf },
        },
        system: { log_level: logLevel, stats_interval: statsInterval },
        events: { screenshot, draw_bbox: drawBbox, draw_roi: drawRoi, draw_tripwire: drawTripwire },
      }
      const res = await api.updateSystemConfig(payload) as Record<string, unknown>
      showToast((res.message as string) || '已保存', true)
      setEditing(false)
      // Refresh config
      const cfg = await api.getSystemConfig()
      setSysConfig(cfg)
      syncFromConfig(cfg)
    } catch (e: unknown) {
      showToast((e as Error).message || '保存失败', false)
    } finally {
      setSaving(false)
    }
  }

  function handleCancel() {
    syncFromConfig(sysConfig)
    setEditing(false)
  }

  const cameras = (stats.cameras || {}) as Record<string, Record<string, number>>
  const healthCams = (health.cameras || {}) as Record<string, Record<string, boolean>>
  const ok = health.status === 'healthy'

  return (
    <div className="p-5 flex flex-col gap-5 overflow-y-auto">
      <div className="flex items-center justify-between">
        <h1 className="text-base font-semibold">系统状态</h1>
        {toast && (
          <span className={`text-xs px-3 py-1.5 rounded-lg ${toast.ok ? 'bg-success-dim text-success' : 'bg-danger-dim text-danger'}`}>
            {toast.msg}
          </span>
        )}
      </div>

      {/* Overview */}
      <div className="grid grid-cols-3 gap-4">
        <Card icon={Activity} label="系统状态" value={ok ? '正常' : '异常'} cls={ok ? 'text-success' : 'text-danger'} bg={ok ? 'bg-success-dim' : 'bg-danger-dim'} />
        <Card icon={Camera} label="摄像头" value={String(stats.camera_count ?? '-')} cls="text-accent" bg="bg-accent-dim" />
        <Card icon={Clock} label="运行时间" value={stats.uptime ? uptime(Date.now() / 1000 - (stats.uptime as number)) : '-'} cls="text-foreground" bg="bg-card" />
      </div>

      {/* Model & System Config */}
      <div className="grid grid-cols-2 gap-4">
        {/* Model config */}
        <div className="bg-card rounded-2xl border border-border p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Box size={14} className="text-accent" />
              <h2 className="text-sm font-medium text-muted-light">模型配置</h2>
            </div>
            {!editing && (
              <button onClick={() => setEditing(true)} className="text-[11px] text-accent hover:text-accent/80 cursor-pointer transition-colors">
                编辑
              </button>
            )}
          </div>
          {editing ? (
            <div className="flex flex-col gap-2.5 text-sm">
              <Field label="检测模型" value={modelPath} onChange={setModelPath} mono />
              <Field label="分析帧率" value={analyzeFps} onChange={v => setAnalyzeFps(Number(v))} type="number" suffix="fps" />
              <Field label="置信度" value={confidence} onChange={v => setConfidence(Number(v))} type="number" step="0.05" />
              <Field label="检测类别" value={classes} onChange={setClasses} placeholder="0, 1, 2, 3" mono />
              <Toggle label="姿态检测" checked={poseEnabled} onChange={setPoseEnabled} />
              {poseEnabled && (
                <>
                  <Field label="姿态模型" value={posePath} onChange={setPosePath} mono />
                  <Field label="姿态置信度" value={poseConf} onChange={v => setPoseConf(Number(v))} type="number" step="0.05" />
                </>
              )}
            </div>
          ) : (
            <dl className="grid grid-cols-[100px_1fr] gap-x-3 gap-y-2 text-sm">
              <dt className="text-muted">检测模型</dt><dd className="font-mono text-xs break-all">{modelPath || '-'}</dd>
              <dt className="text-muted">分析帧率</dt><dd className="font-mono">{analyzeFps} fps</dd>
              <dt className="text-muted">置信度</dt><dd className="font-mono">{confidence}</dd>
              <dt className="text-muted">检测类别</dt><dd className="font-mono text-xs">{classes || '-'}</dd>
              <dt className="text-muted">姿态检测</dt><dd>{poseEnabled ? <span className="text-success text-xs">启用</span> : <span className="text-muted text-xs">禁用</span>}</dd>
              {poseEnabled && <><dt className="text-muted">姿态模型</dt><dd className="font-mono text-xs">{posePath || '-'}</dd></>}
            </dl>
          )}
        </div>

        {/* System & Events config */}
        <div className="bg-card rounded-2xl border border-border p-5">
          <div className="flex items-center gap-2 mb-3">
            <Settings size={14} className="text-accent" />
            <h2 className="text-sm font-medium text-muted-light">系统配置</h2>
          </div>
          {editing ? (
            <div className="flex flex-col gap-2.5 text-sm">
              <div className="flex items-center gap-2">
                <label className="w-[100px] text-muted shrink-0">日志级别</label>
                <select value={logLevel} onChange={e => setLogLevel(e.target.value)}
                  className="flex-1 bg-surface border border-border rounded-lg px-2.5 py-1.5 text-sm text-foreground focus:outline-none focus:border-accent">
                  {['DEBUG', 'INFO', 'WARNING', 'ERROR'].map(l => <option key={l} value={l}>{l}</option>)}
                </select>
              </div>
              <Field label="统计间隔" value={statsInterval} onChange={v => setStatsInterval(Number(v))} type="number" suffix="s" />
              <Toggle label="截图" checked={screenshot} onChange={setScreenshot} />
              <Toggle label="绘制 BBox" checked={drawBbox} onChange={setDrawBbox} />
              <Toggle label="绘制 ROI" checked={drawRoi} onChange={setDrawRoi} />
              <Toggle label="绘制越线" checked={drawTripwire} onChange={setDrawTripwire} />
            </div>
          ) : (
            <dl className="grid grid-cols-[100px_1fr] gap-x-3 gap-y-2 text-sm">
              <dt className="text-muted">日志级别</dt><dd className="font-mono">{logLevel}</dd>
              <dt className="text-muted">统计间隔</dt><dd className="font-mono">{statsInterval}s</dd>
              <dt className="text-muted">截图</dt><dd>{screenshot ? <span className="text-success text-xs">启用</span> : <span className="text-muted text-xs">禁用</span>}</dd>
              <dt className="text-muted">绘制 BBox</dt><dd>{drawBbox ? <span className="text-success text-xs">启用</span> : <span className="text-muted text-xs">禁用</span>}</dd>
              <dt className="text-muted">绘制 ROI</dt><dd>{drawRoi ? <span className="text-success text-xs">启用</span> : <span className="text-muted text-xs">禁用</span>}</dd>
              <dt className="text-muted">绘制越线</dt><dd>{drawTripwire ? <span className="text-success text-xs">启用</span> : <span className="text-muted text-xs">禁用</span>}</dd>
            </dl>
          )}
        </div>
      </div>

      {/* Edit action bar */}
      {editing && (
        <div className="flex items-center gap-3">
          <button onClick={handleSave} disabled={saving}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-accent text-background text-sm font-medium hover:bg-accent/90 transition-colors cursor-pointer disabled:opacity-50">
            {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            {saving ? '保存中...' : '保存配置'}
          </button>
          <button onClick={handleCancel}
            className="px-4 py-2 rounded-xl border border-border text-sm text-muted hover:text-foreground hover:border-border-light transition-colors cursor-pointer">
            取消
          </button>
        </div>
      )}

      {/* Perf table */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 size={14} className="text-muted" />
          <h2 className="text-sm font-medium text-muted-light">摄像头性能</h2>
        </div>
        <div className="rounded-2xl border border-border bg-surface overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-card/40">
              <tr className="text-muted text-left text-[11px] uppercase tracking-wider">
                <th className="px-4 py-3 font-medium">摄像头</th>
                <th className="px-4 py-3 font-medium">状态</th>
                <th className="px-4 py-3 font-medium">FPS</th>
                <th className="px-4 py-3 font-medium">检测 FPS</th>
                <th className="px-4 py-3 font-medium">跳帧率</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(cameras).length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-16">
                    <Cpu size={24} className="mx-auto mb-2 text-muted/15" />
                    <span className="text-xs text-muted">暂无数据</span>
                  </td>
                </tr>
              ) : Object.entries(cameras).map(([id, perf]) => {
                const running = healthCams[id]?.running
                return (
                  <tr key={id} className="border-t border-border/30 hover:bg-card/30 transition-colors">
                    <td className="px-4 py-3 font-medium">{id}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-lg text-[11px] font-medium ${running ? 'bg-success-dim text-success' : 'bg-danger-dim text-danger'}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${running ? 'bg-success' : 'bg-danger'}`} />
                        {running ? '运行中' : '已停止'}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono font-bold text-accent tabular-nums">{perf.fps ?? '-'}</td>
                    <td className="px-4 py-3 font-mono font-bold text-success tabular-nums">{perf.detection_fps ?? '-'}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
                          <div className="h-full bg-accent rounded-full transition-all" style={{ width: `${Math.min(perf.skip_rate ?? 0, 100)}%` }} />
                        </div>
                        <span className="text-muted text-xs font-mono tabular-nums">{perf.skip_rate !== undefined ? `${perf.skip_rate}%` : '-'}</span>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Sub-components ──

function Card({ icon: Icon, label, value, cls, bg }: { icon: typeof Activity; label: string; value: string; cls: string; bg: string }) {
  return (
    <div className="bg-card rounded-2xl border border-border p-5 hover:border-border-light transition-colors cursor-default">
      <div className="flex items-center gap-3 mb-3">
        <span className={`w-8 h-8 rounded-xl ${bg} flex items-center justify-center`}>
          <Icon size={16} className={cls} />
        </span>
        <span className="text-[11px] text-muted uppercase tracking-wider">{label}</span>
      </div>
      <div className={`text-2xl font-bold ${cls}`}>{value}</div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text', mono, placeholder, suffix, step }: {
  label: string; value: string | number; onChange: (v: string) => void
  type?: string; mono?: boolean; placeholder?: string; suffix?: string; step?: string
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="w-[100px] text-muted shrink-0">{label}</label>
      <div className="flex-1 flex items-center gap-1.5">
        <input type={type} value={value} onChange={e => onChange(e.target.value)}
          placeholder={placeholder} step={step}
          className={`flex-1 bg-surface border border-border rounded-lg px-2.5 py-1.5 text-sm text-foreground focus:outline-none focus:border-accent ${mono ? 'font-mono text-xs' : ''}`} />
        {suffix && <span className="text-muted text-xs">{suffix}</span>}
      </div>
    </div>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center gap-2">
      <label className="w-[100px] text-muted shrink-0">{label}</label>
      <button type="button" onClick={() => onChange(!checked)}
        className={`relative w-9 h-5 rounded-full transition-colors cursor-pointer ${checked ? 'bg-accent' : 'bg-border'}`}>
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-4' : ''}`} />
      </button>
      <span className="text-xs text-muted">{checked ? '启用' : '禁用'}</span>
    </div>
  )
}

function uptime(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (h > 24) return `${Math.floor(h / 24)}天 ${h % 24}时`
  return `${h}时 ${m}分`
}

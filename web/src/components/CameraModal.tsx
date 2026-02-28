import { useEffect, useState } from 'react'
import { api } from '../api'
import { X } from 'lucide-react'
import { DrawingCanvas } from './DrawingCanvas'
import type { Point, Tripwire } from './DrawingCanvas'

interface Props {
  mode: 'add' | 'edit'
  camId?: string
  onClose: () => void
  onSaved: () => void
}

const defaultForm = {
  id: '', name: '', url: '', width: 1280, height: 720, fps: 15,
  roi: [] as number[][],
  tripwires: [] as Tripwire[],
  motion: {
    enabled: true, threshold: 40, contour_area: 200, frame_alpha: 0.02,
  },
  rules: {
    intrusion: { enabled: false, confirm_frames: 5, cooldown: 30 },
    tripwire: { enabled: false },
    counting: { enabled: false, window_seconds: 60 },
    anomaly: {
      dwell: { enabled: false, max_seconds: 120, confirm_frames: 5, cooldown: 60 },
      crowd: { enabled: false, max_count: 5, radius: 200, confirm_frames: 5, cooldown: 60 },
      proximity: { enabled: false, min_distance: 50, confirm_frames: 3, cooldown: 30 },
      fight: { enabled: false, proximity_radius: 150, min_speed: 60, min_persons: 2, confirm_frames: 3, cooldown: 30 },
      fall: { enabled: false, ratio_threshold: 1.0, min_ratio_change: 0.5, min_y_drop: 20, confirm_frames: 2, cooldown: 30 },
    },
    alert_types: [] as string[],
  },
}

type FormData = typeof defaultForm

export function CameraModal({ mode, camId, onClose, onSaved }: Props) {
  const [form, setForm] = useState<FormData>(JSON.parse(JSON.stringify(defaultForm)))
  const [saving, setSaving] = useState(false)
  const [tab, setTab] = useState<'basic' | 'draw' | 'rules'>('basic')

  useEffect(() => {
    if (mode === 'edit' && camId) {
      api.getCamera(camId).then(d => {
        // Deep merge with defaults so missing nested keys don't crash
        const base = JSON.parse(JSON.stringify(defaultForm))
        setForm(deepMerge(base, d as Record<string, unknown>) as FormData)
      }).catch(() => {})
    }
  }, [mode, camId])

  const set = (path: string, value: unknown) => {
    setForm(prev => {
      const next = JSON.parse(JSON.stringify(prev))
      const keys = path.split('.')
      let obj = next as Record<string, unknown>
      for (let i = 0; i < keys.length - 1; i++) obj = obj[keys[i]] as Record<string, unknown>
      obj[keys[keys.length - 1]] = value
      return next
    })
  }

  const save = async () => {
    setSaving(true)
    try {
      // Auto-build alert_types from enabled rules
      const payload = JSON.parse(JSON.stringify(form))
      const alertTypes: string[] = []
      if (payload.rules.intrusion?.enabled) alertTypes.push('intrusion')
      if (payload.rules.tripwire?.enabled) alertTypes.push('tripwire')
      if (payload.rules.anomaly?.dwell?.enabled) alertTypes.push('anomaly/dwell')
      if (payload.rules.anomaly?.crowd?.enabled) alertTypes.push('anomaly/crowd')
      if (payload.rules.anomaly?.proximity?.enabled) alertTypes.push('anomaly/proximity')
      if (payload.rules.anomaly?.fight?.enabled) alertTypes.push('anomaly/fight')
      if (payload.rules.anomaly?.fall?.enabled) alertTypes.push('anomaly/fall')
      // Always include presence if any rule is on
      alertTypes.push('presence')
      payload.rules.alert_types = alertTypes

      if (mode === 'add') await api.addCamera(payload)
      else await api.updateCamera(camId!, payload)
      onSaved()
    } catch (e: unknown) { alert((e as Error).message) }
    finally { setSaving(false) }
  }

  const Toggle = ({ label, path }: { label: string; path: string }) => {
    const keys = path.split('.')
    let val: unknown = form
    for (const k of keys) val = (val as Record<string, unknown>)[k]
    return (
      <label className="flex items-center gap-2.5 text-sm cursor-pointer select-none group">
        <span className={`w-8 h-[18px] rounded-full relative transition-colors ${val ? 'bg-accent' : 'bg-border-light'}`}>
          <span className={`absolute top-0.5 w-3.5 h-3.5 rounded-full bg-white transition-all ${val ? 'left-[17px]' : 'left-0.5'}`} />
        </span>
        <input type="checkbox" checked={!!val} onChange={e => set(path, e.target.checked)} className="sr-only" />
        <span className="text-muted-light group-hover:text-foreground transition-colors">{label}</span>
      </label>
    )
  }

  const tabs = [
    { key: 'basic' as const, label: '基本信息' },
    { key: 'draw' as const, label: '区域 & 入口线' },
    { key: 'rules' as const, label: '检测规则' },
  ]

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-start justify-center pt-6 overflow-y-auto" onClick={onClose}>
      <div className="bg-surface rounded-2xl w-[860px] max-w-[95vw] p-6 relative mb-10 border border-border animate-fade-up" onClick={e => e.stopPropagation()}>
        <button onClick={onClose} className="absolute top-4 right-4 w-7 h-7 rounded-lg flex items-center justify-center text-muted hover:text-foreground hover:bg-card cursor-pointer transition-colors">
          <X size={15} />
        </button>
        <h3 className="text-base font-semibold mb-4">{mode === 'add' ? '添加摄像头' : `编辑 ${camId}`}</h3>

        {/* Tabs */}
        <div className="flex gap-1 mb-5 bg-bg-elevated rounded-xl p-1">
          {tabs.map(t => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 py-2 text-sm rounded-lg cursor-pointer transition-colors ${
                tab === t.key ? 'bg-card text-foreground font-medium shadow-sm' : 'text-muted hover:text-foreground'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab: Basic */}
        {tab === 'basic' && (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <Field label="摄像头 ID" value={form.id} onChange={v => set('id', v)} disabled={mode === 'edit'} />
              <Field label="名称" value={form.name} onChange={v => set('name', v)} />
              <div className="col-span-2">
                <Field label="RTSP 地址" value={form.url} onChange={v => set('url', v)} />
              </div>
              <Field label="宽度" value={form.width} onChange={v => set('width', Number(v))} type="number" />
              <Field label="高度" value={form.height} onChange={v => set('height', Number(v))} type="number" />
              <Field label="FPS" value={form.fps} onChange={v => set('fps', Number(v))} type="number" />
            </div>
            {/* Motion detection */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-muted-light font-medium uppercase tracking-wider">运动检测</span>
                <Toggle label="启用" path="motion.enabled" />
              </div>
              {form.motion.enabled && (
                <div className="grid grid-cols-3 gap-3 text-sm bg-bg-elevated rounded-xl p-3">
                  <Field label="阈值" value={form.motion.threshold} onChange={v => set('motion.threshold', Number(v))} type="number" />
                  <Field label="最小轮廓面积" value={form.motion.contour_area} onChange={v => set('motion.contour_area', Number(v))} type="number" />
                  <Field label="帧混合系数" value={form.motion.frame_alpha} onChange={v => set('motion.frame_alpha', Number(v))} type="number" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab: Draw ROI & Tripwires */}
        {tab === 'draw' && (
          <DrawingCanvas
            cameraId={mode === 'edit' ? camId! : form.id}
            width={form.width || 1280}
            height={form.height || 720}
            roi={(form.roi || []) as Point[]}
            tripwires={(form.tripwires || []) as Tripwire[]}
            onRoiChange={pts => set('roi', pts)}
            onTripwiresChange={tws => set('tripwires', tws)}
          />
        )}

        {/* Tab: Rules */}
        {tab === 'rules' && (
          <div className="space-y-4">
            {/* Intrusion */}
            <RuleSection>
              <Toggle label="入侵检测" path="rules.intrusion.enabled" />
              {form.rules.intrusion.enabled && (
                <div className="grid grid-cols-2 gap-3 mt-2 text-sm">
                  <Field label="确认帧数" value={form.rules.intrusion.confirm_frames} onChange={v => set('rules.intrusion.confirm_frames', Number(v))} type="number" />
                  <Field label="冷却时间(秒)" value={form.rules.intrusion.cooldown} onChange={v => set('rules.intrusion.cooldown', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>

            {/* Tripwire */}
            <RuleSection>
              <Toggle label="越线检测" path="rules.tripwire.enabled" />
            </RuleSection>

            {/* Counting */}
            <RuleSection>
              <Toggle label="流量计数" path="rules.counting.enabled" />
              {form.rules.counting.enabled && (
                <div className="grid grid-cols-1 gap-3 mt-2 text-sm max-w-[200px]">
                  <Field label="窗口时间(秒)" value={form.rules.counting.window_seconds} onChange={v => set('rules.counting.window_seconds', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>

            {/* Anomaly: Dwell */}
            <RuleSection>
              <Toggle label="滞留检测" path="rules.anomaly.dwell.enabled" />
              {form.rules.anomaly.dwell.enabled && (
                <div className="grid grid-cols-3 gap-3 mt-2 text-sm">
                  <Field label="最大停留(秒)" value={form.rules.anomaly.dwell.max_seconds} onChange={v => set('rules.anomaly.dwell.max_seconds', Number(v))} type="number" />
                  <Field label="确认帧数" value={form.rules.anomaly.dwell.confirm_frames} onChange={v => set('rules.anomaly.dwell.confirm_frames', Number(v))} type="number" />
                  <Field label="冷却时间(秒)" value={form.rules.anomaly.dwell.cooldown} onChange={v => set('rules.anomaly.dwell.cooldown', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>

            {/* Anomaly: Crowd */}
            <RuleSection>
              <Toggle label="聚集检测" path="rules.anomaly.crowd.enabled" />
              {form.rules.anomaly.crowd.enabled && (
                <div className="grid grid-cols-2 gap-3 mt-2 text-sm">
                  <Field label="最大人数" value={form.rules.anomaly.crowd.max_count} onChange={v => set('rules.anomaly.crowd.max_count', Number(v))} type="number" />
                  <Field label="半径(px)" value={form.rules.anomaly.crowd.radius} onChange={v => set('rules.anomaly.crowd.radius', Number(v))} type="number" />
                  <Field label="确认帧数" value={form.rules.anomaly.crowd.confirm_frames} onChange={v => set('rules.anomaly.crowd.confirm_frames', Number(v))} type="number" />
                  <Field label="冷却时间(秒)" value={form.rules.anomaly.crowd.cooldown} onChange={v => set('rules.anomaly.crowd.cooldown', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>

            {/* Anomaly: Proximity */}
            <RuleSection>
              <Toggle label="人车过近" path="rules.anomaly.proximity.enabled" />
              {form.rules.anomaly.proximity.enabled && (
                <div className="grid grid-cols-3 gap-3 mt-2 text-sm">
                  <Field label="最小距离(px)" value={form.rules.anomaly.proximity.min_distance} onChange={v => set('rules.anomaly.proximity.min_distance', Number(v))} type="number" />
                  <Field label="确认帧数" value={form.rules.anomaly.proximity.confirm_frames} onChange={v => set('rules.anomaly.proximity.confirm_frames', Number(v))} type="number" />
                  <Field label="冷却时间(秒)" value={form.rules.anomaly.proximity.cooldown} onChange={v => set('rules.anomaly.proximity.cooldown', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>

            {/* Anomaly: Fight */}
            <RuleSection>
              <Toggle label="打架检测" path="rules.anomaly.fight.enabled" />
              {form.rules.anomaly.fight.enabled && (
                <div className="grid grid-cols-3 gap-3 mt-2 text-sm">
                  <Field label="接近半径(px)" value={form.rules.anomaly.fight.proximity_radius} onChange={v => set('rules.anomaly.fight.proximity_radius', Number(v))} type="number" />
                  <Field label="最小速度" value={form.rules.anomaly.fight.min_speed} onChange={v => set('rules.anomaly.fight.min_speed', Number(v))} type="number" />
                  <Field label="最少人数" value={form.rules.anomaly.fight.min_persons} onChange={v => set('rules.anomaly.fight.min_persons', Number(v))} type="number" />
                  <Field label="确认帧数" value={form.rules.anomaly.fight.confirm_frames} onChange={v => set('rules.anomaly.fight.confirm_frames', Number(v))} type="number" />
                  <Field label="冷却时间(秒)" value={form.rules.anomaly.fight.cooldown} onChange={v => set('rules.anomaly.fight.cooldown', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>

            {/* Anomaly: Fall */}
            <RuleSection>
              <Toggle label="跌倒检测" path="rules.anomaly.fall.enabled" />
              {form.rules.anomaly.fall.enabled && (
                <div className="grid grid-cols-3 gap-3 mt-2 text-sm">
                  <Field label="比例阈值" value={form.rules.anomaly.fall.ratio_threshold} onChange={v => set('rules.anomaly.fall.ratio_threshold', Number(v))} type="number" />
                  <Field label="最小比例变化" value={form.rules.anomaly.fall.min_ratio_change} onChange={v => set('rules.anomaly.fall.min_ratio_change', Number(v))} type="number" />
                  <Field label="最小Y下降(px)" value={form.rules.anomaly.fall.min_y_drop} onChange={v => set('rules.anomaly.fall.min_y_drop', Number(v))} type="number" />
                  <Field label="确认帧数" value={form.rules.anomaly.fall.confirm_frames} onChange={v => set('rules.anomaly.fall.confirm_frames', Number(v))} type="number" />
                  <Field label="冷却时间(秒)" value={form.rules.anomaly.fall.cooldown} onChange={v => set('rules.anomaly.fall.cooldown', Number(v))} type="number" />
                </div>
              )}
            </RuleSection>
          </div>
        )}

        <div className="flex justify-end gap-3 mt-6">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded-xl text-sm cursor-pointer hover:bg-card transition-colors">取消</button>
          <button onClick={save} disabled={saving} className="px-5 py-2 bg-accent text-white rounded-xl text-sm font-medium cursor-pointer hover:bg-accent-light transition-colors disabled:opacity-40">
            {saving ? '保存中…' : '保存'}
          </button>
        </div>
      </div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text', disabled = false }: {
  label: string; value: string | number; onChange: (v: string) => void; type?: string; disabled?: boolean
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[11px] text-muted">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
        className="bg-bg-elevated border border-border rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-accent disabled:opacity-40 transition-colors"
      />
    </div>
  )
}

function RuleSection({ children }: { children: React.ReactNode }) {
  return <div className="bg-bg-elevated rounded-xl p-3 border border-border/50">{children}</div>
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function deepMerge(base: Record<string, any>, override: Record<string, any>): Record<string, any> {
  const result = { ...base }
  for (const key of Object.keys(override)) {
    if (
      override[key] !== null &&
      typeof override[key] === 'object' &&
      !Array.isArray(override[key]) &&
      typeof base[key] === 'object' &&
      base[key] !== null &&
      !Array.isArray(base[key])
    ) {
      result[key] = deepMerge(base[key], override[key])
    } else {
      result[key] = override[key]
    }
  }
  return result
}

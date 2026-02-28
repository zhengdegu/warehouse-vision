import { useEffect, useState, useRef, useCallback } from 'react'
import { api } from '../api'
import { useWebSocket } from '../hooks/useWebSocket'
import { AnnotationEditor } from '../components/AnnotationEditor'
import { Database, ImageIcon, Tag, Play, Box, Trash2, Upload, Wand2, Plus, X, Rocket, ChevronLeft, ChevronRight } from 'lucide-react'

type Tab = 'datasets' | 'samples' | 'annotations' | 'jobs' | 'models'
type R = Record<string, unknown>

const tabs: { id: Tab; label: string; icon: typeof Database }[] = [
  { id: 'datasets', label: '数据集', icon: Database },
  { id: 'samples', label: '样本', icon: ImageIcon },
  { id: 'annotations', label: '标注', icon: Tag },
  { id: 'jobs', label: '训练任务', icon: Play },
  { id: 'models', label: '模型', icon: Box },
]

export function TrainingPage() {
  const [tab, setTab] = useState<Tab>('datasets')
  return (
    <div className="flex h-full">
      <div className="w-40 bg-surface border-r border-border shrink-0 py-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            className={`w-full flex items-center gap-2 px-4 py-2.5 text-sm cursor-pointer border-l-2 transition-colors ${tab === id ? 'border-accent text-accent bg-accent-dim' : 'border-transparent text-muted hover:text-muted-light hover:bg-card/40'}`}>
            <Icon size={14} /> {label}
          </button>
        ))}
      </div>
      <div className="flex-1 p-5 overflow-y-auto">
        {tab === 'datasets' && <DatasetsTab />}
        {tab === 'samples' && <SamplesTab />}
        {tab === 'annotations' && <AnnotationsTab />}
        {tab === 'jobs' && <JobsTab />}
        {tab === 'models' && <ModelsTab />}
      </div>
    </div>
  )
}

/* ── Datasets ── */
function DatasetsTab() {
  const [datasets, setDatasets] = useState<R[]>([])
  const [stats, setStats] = useState<Record<string, R>>({})
  const [name, setName] = useState('')
  const [classes, setClasses] = useState('')

  const load = useCallback(async () => {
    const d = await api.getDatasets().catch(() => ({ items: [] }))
    const items = d.items || []
    setDatasets(items)
    const s: Record<string, R> = {}
    for (const ds of items) {
      try { s[ds.name as string] = await api.getDatasetStats(ds.name as string) } catch {}
    }
    setStats(s)
  }, [])

  useEffect(() => { load() }, [load])

  const create = async () => {
    if (!name) return
    await api.createDataset({ name, classes: classes.split(',').map(s => s.trim()).filter(Boolean) }).catch(e => alert(e.message))
    setName(''); setClasses(''); load()
  }

  const del = async (n: string) => {
    if (!confirm(`确定删除数据集 ${n}？所有样本和标注将被删除`)) return
    await api.deleteDataset(n).catch(e => alert(e.message)); load()
  }

  return (
    <div>
      <h2 className="text-base font-semibold mb-4">数据集管理</h2>
      <div className="bg-card rounded-xl border border-border p-4 mb-4">
        <div className="flex gap-3 items-end">
          <FField label="名称" value={name} onChange={setName} placeholder="warehouse_v1" />
          <FField label="类别 (逗号分隔)" value={classes} onChange={setClasses} placeholder="person,forklift" className="w-56" />
          <button onClick={create} className="px-4 py-1.5 bg-accent text-white rounded-lg text-sm cursor-pointer hover:bg-accent-light transition-colors shrink-0">创建</button>
        </div>
      </div>
      <Table cols={['名称', '类别', '样本数', '已标注', '未标注', '']} empty="暂无数据集"
        rows={datasets.map((ds, i) => {
          const s = stats[ds.name as string] || {}
          return (
            <tr key={i} className="border-t border-border/30 hover:bg-card/30 transition-colors">
              <td className="px-4 py-2.5 font-medium">{ds.name as string}</td>
              <td className="px-4 py-2.5 text-muted">{((ds.classes as string[]) || []).join(', ')}</td>
              <td className="px-4 py-2.5 font-mono">{s.total_samples as number ?? '-'}</td>
              <td className="px-4 py-2.5 font-mono text-success">{s.annotated_count as number ?? '-'}</td>
              <td className="px-4 py-2.5 font-mono text-warning">{s.unannotated_count as number ?? '-'}</td>
              <td className="px-4 py-2.5">
                <button onClick={() => del(ds.name as string)} className="p-1.5 rounded-lg hover:bg-danger-dim text-muted hover:text-danger cursor-pointer transition-colors">
                  <Trash2 size={13} />
                </button>
              </td>
            </tr>
          )
        })}
      />
    </div>
  )
}

/* ── Samples ── */
function SamplesTab() {
  const [datasets, setDatasets] = useState<R[]>([])
  const [dsFilter, setDsFilter] = useState('')
  const [samples, setSamples] = useState<R[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [uploading, setUploading] = useState(false)
  const [editorIdx, setEditorIdx] = useState<number | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const pageSize = 20

  useEffect(() => { api.getDatasets().then(d => setDatasets(d.items || [])).catch(() => {}) }, [])

  const load = useCallback(() => {
    const p = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
    if (dsFilter) p.set('dataset_name', dsFilter)
    api.getSamples(p.toString()).then(d => { setSamples(d.items || []); setTotal(d.total || 0) }).catch(() => {})
  }, [page, dsFilter])

  useEffect(() => { load() }, [load])

  const upload = async (files: FileList | null) => {
    if (!files || files.length === 0 || !dsFilter) { alert('请先选择数据集'); return }
    setUploading(true)
    try { await api.uploadSamples(Array.from(files), dsFilter); load() }
    catch (e) { alert((e as Error).message) }
    finally { setUploading(false) }
  }

  const del = async (id: string) => {
    await api.deleteSample(id).catch(e => alert(e.message)); load()
  }

  // Get classes for the filtered dataset
  const dsClasses = (datasets.find(d => d.name === dsFilter) as R | undefined)
  const classes = dsClasses ? (dsClasses.classes as string[]) || [] : []

  const totalPages = Math.ceil(total / pageSize)

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold">样本管理</h2>
        <div className="flex items-center gap-3">
          <select value={dsFilter} onChange={e => { setDsFilter(e.target.value); setPage(1) }}
            className="bg-card border border-border rounded-xl px-3 py-2 text-sm cursor-pointer focus:outline-none focus:border-accent">
            <option value="">全部数据集</option>
            {datasets.map(d => <option key={d.name as string} value={d.name as string}>{d.name as string}</option>)}
          </select>
          <input ref={fileRef} type="file" multiple accept=".jpg,.jpeg,.png" className="hidden" onChange={e => upload(e.target.files)} />
          <button onClick={() => fileRef.current?.click()} disabled={uploading || !dsFilter}
            className="flex items-center gap-1.5 px-4 py-2 bg-accent text-white rounded-xl text-sm cursor-pointer hover:bg-accent-light transition-colors disabled:opacity-40">
            <Upload size={13} /> {uploading ? '上传中…' : '上传样本'}
          </button>
        </div>
      </div>
      <Table cols={['ID', '文件名', '数据集', '已标注', '上传时间', '']} empty="暂无样本"
        rows={samples.map((s, i) => (
          <tr key={i} className="border-t border-border/30 hover:bg-card/30 transition-colors">
            <td className="px-4 py-2.5 font-mono text-xs text-muted">{((s.sample_id as string) || '').slice(0, 10)}…</td>
            <td className="px-4 py-2.5 text-sm">
              <button onClick={() => { if (!dsFilter) { alert('请先选择数据集'); return }; setEditorIdx(i) }} className="text-accent hover:underline cursor-pointer">{s.filename as string}</button>
            </td>
            <td className="px-4 py-2.5 text-muted text-sm">{s.dataset_name as string}</td>
            <td className="px-4 py-2.5">{s.annotated ? <span className="text-success text-xs">✓ 已标注</span> : <span className="text-muted text-xs">未标注</span>}</td>
            <td className="px-4 py-2.5 text-muted text-xs font-mono">{s.upload_time ? new Date(s.upload_time as string).toLocaleString('zh-CN') : '-'}</td>
            <td className="px-4 py-2.5 flex gap-1">
              <button onClick={() => { if (!dsFilter) { alert('请先选择数据集'); return }; setEditorIdx(i) }} className="p-1.5 rounded-lg hover:bg-accent-dim text-muted hover:text-accent cursor-pointer transition-colors" title="标注">
                <Tag size={13} />
              </button>
              <button onClick={() => del(s.sample_id as string)} className="p-1.5 rounded-lg hover:bg-danger-dim text-muted hover:text-danger cursor-pointer transition-colors">
                <Trash2 size={13} />
              </button>
            </td>
          </tr>
        ))}
      />
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-4 text-xs">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1} className="p-1.5 rounded-lg hover:bg-card disabled:opacity-20 cursor-pointer"><ChevronLeft size={15} /></button>
          <span className="px-2 py-1 bg-card rounded-lg font-mono">{page} / {totalPages}</span>
          <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages} className="p-1.5 rounded-lg hover:bg-card disabled:opacity-20 cursor-pointer"><ChevronRight size={15} /></button>
        </div>
      )}

      {editorIdx !== null && dsFilter && classes.length > 0 && (
        <AnnotationEditor
          samples={samples.map(s => ({
            sample_id: s.sample_id as string,
            filename: s.filename as string,
            dataset_name: s.dataset_name as string,
            file_path: s.file_path as string,
            annotated: !!s.annotated,
          }))}
          initialIndex={editorIdx}
          classes={classes}
          onClose={() => setEditorIdx(null)}
          onSaved={load}
        />
      )}
    </div>
  )
}

/* ── Annotations (auto-annotate) ── */
function AnnotationsTab() {
  const [datasets, setDatasets] = useState<R[]>([])
  const [dsName, setDsName] = useState('')
  const [modelPath, setModelPath] = useState('yolo26m.pt')
  const [conf, setConf] = useState('0.5')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<R | null>(null)

  useEffect(() => { api.getDatasets().then(d => setDatasets(d.items || [])).catch(() => {}) }, [])

  const run = async () => {
    if (!dsName) { alert('请选择数据集'); return }
    setRunning(true); setResult(null)
    try {
      const r = await api.autoAnnotate(dsName, { model_path: modelPath, confidence_threshold: parseFloat(conf) })
      setResult(r as R)
    } catch (e) { alert((e as Error).message) }
    finally { setRunning(false) }
  }

  return (
    <div>
      <h2 className="text-base font-semibold mb-4">自动标注</h2>
      <div className="bg-card rounded-xl border border-border p-5 max-w-lg">
        <p className="text-sm text-muted mb-4">使用现有 YOLO 模型对数据集中未标注的样本自动生成标注。</p>
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-muted">数据集</label>
            <select value={dsName} onChange={e => setDsName(e.target.value)}
              className="bg-bg-elevated border border-border rounded-xl px-3 py-2 text-sm cursor-pointer focus:outline-none focus:border-accent">
              <option value="">选择数据集</option>
              {datasets.map(d => <option key={d.name as string} value={d.name as string}>{d.name as string}</option>)}
            </select>
          </div>
          <FField label="模型路径" value={modelPath} onChange={setModelPath} placeholder="yolov8n.pt" />
          <FField label="置信度阈值" value={conf} onChange={setConf} placeholder="0.5" />
          <button onClick={run} disabled={running}
            className="flex items-center justify-center gap-2 px-4 py-2.5 bg-accent text-white rounded-xl text-sm cursor-pointer hover:bg-accent-light transition-colors disabled:opacity-40 mt-1">
            <Wand2 size={14} /> {running ? '标注中…' : '开始自动标注'}
          </button>
        </div>
        {result && (
          <div className="mt-4 bg-bg-elevated rounded-xl p-4 text-sm">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div><div className="text-lg font-bold text-success">{result.success_count as number}</div><div className="text-[11px] text-muted">成功</div></div>
              <div><div className="text-lg font-bold text-danger">{result.failed_count as number}</div><div className="text-[11px] text-muted">失败</div></div>
              <div><div className="text-lg font-bold text-muted-light">{result.skipped_count as number}</div><div className="text-[11px] text-muted">跳过</div></div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Jobs ── */
function JobsTab() {
  const [jobs, setJobs] = useState<R[]>([])
  const [datasets, setDatasets] = useState<R[]>([])
  const [models, setModels] = useState<R[]>([])
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState({ dataset_name: '', epochs: 50, batch_size: 16, image_size: 640, base_model: 'yolov8n.pt' })
  const [iterForm, setIterForm] = useState({ dataset_name: '', parent_model_id: '', epochs: 50, batch_size: 16, image_size: 640 })
  const [mode, setMode] = useState<'new' | 'iterate'>('new')
  const { subscribe } = useWebSocket()

  const load = useCallback(() => {
    api.getJobs('page=1&page_size=50').then(d => setJobs(d.items || [])).catch(() => {})
  }, [])

  useEffect(() => { load(); api.getDatasets().then(d => setDatasets(d.items || [])).catch(() => {}); api.getModels().then(d => setModels(d.items || [])).catch(() => {}) }, [load])

  // WebSocket: 实时接收训练进度
  useEffect(() => subscribe((evt) => {
    if (evt.type !== 'training_progress') return
    setJobs(prev => {
      const idx = prev.findIndex(j => j.job_id === evt.job_id)
      if (idx === -1) return prev
      const updated = [...prev]
      updated[idx] = {
        ...updated[idx],
        current_epoch: evt.current_epoch,
        epochs: evt.total_epochs ?? updated[idx].epochs,
        best_map50: evt.best_map50 ?? updated[idx].best_map50,
        status: evt.status ?? updated[idx].status,
        ...(evt.train_loss !== undefined ? { _live_loss: evt.train_loss } : {}),
        ...(evt.map50 !== undefined ? { _live_map50: evt.map50 } : {}),
        ...(evt.output_model_id ? { output_model_id: evt.output_model_id } : {}),
      }
      return updated
    })
    // 训练完成或失败时刷新完整列表
    if (evt.status === 'completed' || evt.status === 'failed' || evt.status === 'cancelled') {
      setTimeout(load, 500)
    }
  }), [subscribe, load])

  // Fallback: 轮询刷新（仅在无 WS 进度时）
  useEffect(() => {
    const hasRunning = jobs.some(j => j.status === 'running' || j.status === 'pending')
    if (!hasRunning) return
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [jobs, load])

  const create = async () => {
    try {
      if (mode === 'new') await api.createJob(form)
      else await api.createIterationJob(iterForm)
      setShowCreate(false); load()
    } catch (e) { alert((e as Error).message) }
  }

  const cancel = async (id: string) => {
    if (!confirm('确定取消此任务？')) return
    await api.cancelJob(id).catch(e => alert(e.message)); load()
  }

  const statusCls: Record<string, string> = { pending: 'bg-accent-dim text-accent', running: 'bg-info-dim text-info', completed: 'bg-success-dim text-success', failed: 'bg-danger-dim text-danger', cancelled: 'bg-card text-muted' }
  const statusLabel: Record<string, string> = { pending: '等待中', running: '训练中', completed: '已完成', failed: '失败', cancelled: '已取消' }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold">训练任务</h2>
        <button onClick={() => setShowCreate(true)} className="flex items-center gap-1.5 px-4 py-2 bg-accent text-white rounded-xl text-sm cursor-pointer hover:bg-accent-light transition-colors">
          <Plus size={13} /> 创建任务
        </button>
      </div>
      <Table cols={['任务ID', '数据集', '状态', '进度', 'Loss', 'mAP50', '']} empty="暂无任务"
        rows={jobs.map((j, i) => {
          const st = (j.status as string) || 'pending'
          const curEpoch = (j.current_epoch as number) || 0
          const totalEpochs = (j.epochs as number) || 1
          const pct = curEpoch / totalEpochs * 100
          const canCancel = st === 'pending' || st === 'running'
          const liveLoss = j._live_loss as number | undefined
          const bestMap = (j.best_map50 as number) || 0
          return (
            <tr key={i} className="border-t border-border/30 hover:bg-card/30 transition-colors">
              <td className="px-4 py-2.5 text-muted font-mono text-xs">{((j.job_id as string) || '').slice(0, 8)}…</td>
              <td className="px-4 py-2.5">{j.dataset_name as string}</td>
              <td className="px-4 py-2.5"><span className={`px-2 py-0.5 rounded-lg text-[11px] font-medium ${statusCls[st] || ''}`}>{statusLabel[st] || st}</span></td>
              <td className="px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <div className="w-20 h-1.5 bg-border rounded-full overflow-hidden"><div className={`h-full rounded-full transition-all ${st === 'running' ? 'bg-accent animate-pulse' : 'bg-accent'}`} style={{ width: `${pct}%` }} /></div>
                  <span className="text-[11px] text-muted font-mono">{curEpoch}/{totalEpochs}</span>
                </div>
              </td>
              <td className="px-4 py-2.5 text-warning font-mono text-xs">{liveLoss !== undefined && liveLoss > 0 ? liveLoss.toFixed(3) : '-'}</td>
              <td className="px-4 py-2.5 text-success font-mono font-bold">{bestMap > 0 ? `${(bestMap * 100).toFixed(1)}%` : '-'}</td>
              <td className="px-4 py-2.5">
                {canCancel && <button onClick={() => cancel(j.job_id as string)} className="p-1.5 rounded-lg hover:bg-danger-dim text-muted hover:text-danger cursor-pointer transition-colors"><X size={13} /></button>}
              </td>
            </tr>
          )
        })}
      />

      {/* Create modal */}
      {showCreate && (
        <Modal title="创建训练任务" onClose={() => setShowCreate(false)}>
          <div className="flex gap-1 mb-4 bg-bg-elevated rounded-xl p-1">
            <button onClick={() => setMode('new')} className={`flex-1 py-2 text-sm rounded-lg cursor-pointer transition-colors ${mode === 'new' ? 'bg-card text-foreground font-medium shadow-sm' : 'text-muted'}`}>新训练</button>
            <button onClick={() => setMode('iterate')} className={`flex-1 py-2 text-sm rounded-lg cursor-pointer transition-colors ${mode === 'iterate' ? 'bg-card text-foreground font-medium shadow-sm' : 'text-muted'}`}>迭代训练</button>
          </div>
          {mode === 'new' ? (
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1"><label className="text-[11px] text-muted">数据集</label>
                <select value={form.dataset_name} onChange={e => setForm({ ...form, dataset_name: e.target.value })} className="bg-bg-elevated border border-border rounded-xl px-3 py-2 text-sm">
                  <option value="">选择</option>{datasets.map(d => <option key={d.name as string} value={d.name as string}>{d.name as string}</option>)}
                </select></div>
              <FField label="Epochs" value={String(form.epochs)} onChange={v => setForm({ ...form, epochs: Number(v) })} />
              <FField label="Batch Size" value={String(form.batch_size)} onChange={v => setForm({ ...form, batch_size: Number(v) })} />
              <FField label="Image Size" value={String(form.image_size)} onChange={v => setForm({ ...form, image_size: Number(v) })} />
              <FField label="Base Model" value={form.base_model} onChange={v => setForm({ ...form, base_model: v })} />
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1"><label className="text-[11px] text-muted">数据集</label>
                <select value={iterForm.dataset_name} onChange={e => setIterForm({ ...iterForm, dataset_name: e.target.value })} className="bg-bg-elevated border border-border rounded-xl px-3 py-2 text-sm">
                  <option value="">选择</option>{datasets.map(d => <option key={d.name as string} value={d.name as string}>{d.name as string}</option>)}
                </select></div>
              <div className="flex flex-col gap-1"><label className="text-[11px] text-muted">父模型</label>
                <select value={iterForm.parent_model_id} onChange={e => setIterForm({ ...iterForm, parent_model_id: e.target.value })} className="bg-bg-elevated border border-border rounded-xl px-3 py-2 text-sm">
                  <option value="">选择</option>{models.map(m => <option key={m.model_id as string} value={m.model_id as string}>{m.version as string} - {m.dataset_name as string}</option>)}
                </select></div>
              <FField label="Epochs" value={String(iterForm.epochs)} onChange={v => setIterForm({ ...iterForm, epochs: Number(v) })} />
            </div>
          )}
          <button onClick={create} className="w-full mt-4 py-2.5 bg-accent text-white rounded-xl text-sm cursor-pointer hover:bg-accent-light transition-colors font-medium">
            <Play size={14} className="inline mr-1.5" />开始训练
          </button>
        </Modal>
      )}
    </div>
  )
}

/* ── Models ── */
function ModelsTab() {
  const [models, setModels] = useState<R[]>([])
  const load = useCallback(() => { api.getModels().then(d => setModels(d.items || [])).catch(() => {}) }, [])
  useEffect(() => { load() }, [load])

  const publish = async (id: string) => {
    if (!confirm('发布此模型到生产环境？将替换当前使用的模型。')) return
    try { await api.publishModel(id); load() } catch (e) { alert((e as Error).message) }
  }
  const del = async (id: string) => {
    if (!confirm('确定删除此模型？')) return
    try { await api.deleteModel(id); load() } catch (e) { alert((e as Error).message) }
  }

  return (
    <div>
      <h2 className="text-base font-semibold mb-4">模型管理</h2>
      <Table cols={['版本', '模型ID', '数据集', 'mAP50', '状态', '']} empty="暂无模型"
        rows={models.map((m, i) => {
          const metrics = (m.metrics as Record<string, number>) || {}
          return (
            <tr key={i} className="border-t border-border/30 hover:bg-card/30 transition-colors">
              <td className="px-4 py-2.5 font-medium">{m.version as string}</td>
              <td className="px-4 py-2.5 text-muted font-mono text-xs">{((m.model_id as string) || '').slice(0, 12)}…</td>
              <td className="px-4 py-2.5">{m.dataset_name as string}</td>
              <td className="px-4 py-2.5 text-success font-mono font-bold">{metrics.map50 !== undefined ? `${(metrics.map50 * 100).toFixed(1)}%` : '-'}</td>
              <td className="px-4 py-2.5">
                {m.published ? <span className="px-2 py-0.5 rounded-lg text-[11px] font-medium bg-success-dim text-success">生产中</span>
                  : <span className="text-muted text-xs">未发布</span>}
              </td>
              <td className="px-4 py-2.5 flex gap-1">
                {!m.published && (
                  <button onClick={() => publish(m.model_id as string)} className="p-1.5 rounded-lg hover:bg-success-dim text-muted hover:text-success cursor-pointer transition-colors" title="发布">
                    <Rocket size={13} />
                  </button>
                )}
                <button onClick={() => del(m.model_id as string)} className="p-1.5 rounded-lg hover:bg-danger-dim text-muted hover:text-danger cursor-pointer transition-colors" title="删除">
                  <Trash2 size={13} />
                </button>
              </td>
            </tr>
          )
        })}
      />
    </div>
  )
}

/* ── Shared components ── */
function Table({ cols, rows, empty }: { cols: string[]; rows: React.ReactNode[]; empty: string }) {
  return (
    <div className="rounded-2xl border border-border bg-surface overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-card/40">
          <tr className="text-muted text-left text-[11px] uppercase tracking-wider">
            {cols.map(c => <th key={c} className="px-4 py-3 font-medium">{c}</th>)}
          </tr>
        </thead>
        <tbody>{rows.length === 0 ? <tr><td colSpan={cols.length} className="text-center text-muted py-12 text-xs">{empty}</td></tr> : rows}</tbody>
      </table>
    </div>
  )
}

function FField({ label, value, onChange, placeholder, className }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; className?: string }) {
  return (
    <div className={`flex flex-col gap-1 ${className || ''}`}>
      <label className="text-[11px] text-muted">{label}</label>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        className="bg-bg-elevated border border-border rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-accent" />
    </div>
  )
}

function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-start justify-center pt-16" onClick={onClose}>
      <div className="bg-surface rounded-2xl w-[480px] max-w-[95vw] p-6 border border-border animate-fade-up" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold">{title}</h3>
          <button onClick={onClose} className="w-7 h-7 rounded-lg flex items-center justify-center text-muted hover:text-foreground hover:bg-card cursor-pointer"><X size={15} /></button>
        </div>
        {children}
      </div>
    </div>
  )
}

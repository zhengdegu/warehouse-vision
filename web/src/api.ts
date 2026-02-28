const BASE = ''

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || err.detail || res.statusText)
  }
  return res.json()
}

const json = (data: unknown) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })

export const api = {
  // ── Cameras ──
  getCameras: () => request<{ id: string; name: string }[]>('/api/cameras'),
  getCamera: (id: string) => request<Record<string, unknown>>(`/api/cameras/${id}`),
  addCamera: (data: unknown) => request('/api/cameras', json(data)),
  updateCamera: (id: string, data: unknown) => request(`/api/cameras/${id}`, { ...json(data), method: 'PUT' }),
  deleteCamera: (id: string) => request(`/api/cameras/${id}`, { method: 'DELETE' }),

  // ── Counts & Events ──
  getCounts: () => request<Record<string, Record<string, number>>>('/api/counts'),
  getEvents: (params: string) => request<{ items: Record<string, unknown>[]; total: number }>(`/api/events?${params}`),
  getEventSummary: (params?: string) => request<Record<string, unknown>>(`/api/events/summary${params ? '?' + params : ''}`),

  // ── System ──
  getSystemStats: () => request<Record<string, unknown>>('/api/system/stats'),
  getSystemHealth: () => request<Record<string, unknown>>('/api/system/health'),
  getSystemConfig: () => request<Record<string, unknown>>('/api/system/config'),
  updateSystemConfig: (data: Record<string, unknown>) => request<Record<string, unknown>>('/api/system/config', { ...json(data), method: 'PUT' }),

  // ── Training: Datasets ──
  getDatasets: () => request<{ items: Record<string, unknown>[] }>('/api/training/datasets'),
  createDataset: (data: { name: string; classes: string[] }) => request('/api/training/datasets', json(data)),
  deleteDataset: (name: string) => request(`/api/training/datasets/${name}`, { method: 'DELETE' }),
  getDatasetStats: (name: string) => request<Record<string, unknown>>(`/api/training/datasets/${name}/stats`),

  // ── Training: Samples ──
  getSamples: (params: string) => request<{ items: Record<string, unknown>[]; total: number; total_pages: number }>(`/api/training/samples?${params}`),
  uploadSamples: (files: File[], datasetName?: string) => {
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    const qs = datasetName ? `?dataset_name=${encodeURIComponent(datasetName)}` : ''
    return request<{ items: Record<string, unknown>[] }>(`/api/training/samples/upload${qs}`, { method: 'POST', body: fd })
  },
  deleteSample: (id: string) => request(`/api/training/samples/${id}`, { method: 'DELETE' }),

  // ── Training: Annotations ──
  getAnnotations: (sampleId: string) => request<{ annotations: Record<string, unknown>[] }>(`/api/training/samples/${sampleId}/annotations`),
  saveAnnotations: (sampleId: string, data: unknown) => request(`/api/training/samples/${sampleId}/annotations`, { ...json(data), method: 'PUT' }),
  autoAnnotate: (datasetName: string, data: unknown) => request(`/api/training/datasets/${datasetName}/auto-annotate`, json(data)),

  // ── Training: Jobs ──
  getJobs: (params?: string) => request<{ items: Record<string, unknown>[] }>(`/api/training/jobs${params ? '?' + params : ''}`),
  createJob: (data: unknown) => request('/api/training/jobs', json(data)),
  createIterationJob: (data: unknown) => request('/api/training/jobs/iterate', json(data)),
  cancelJob: (id: string) => request(`/api/training/jobs/${id}/cancel`, { method: 'POST' }),

  // ── Training: Models ──
  getModels: () => request<{ items: Record<string, unknown>[] }>('/api/training/models'),
  publishModel: (id: string) => request(`/api/training/models/${id}/publish`, { method: 'POST' }),
  deleteModel: (id: string) => request(`/api/training/models/${id}`, { method: 'DELETE' }),
}

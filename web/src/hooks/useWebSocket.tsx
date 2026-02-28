import { createContext, useContext, useEffect, useRef, useState, useCallback, type ReactNode } from 'react'

type EventData = Record<string, unknown>
type Listener = (event: EventData) => void

interface WSContext {
  connected: boolean
  subscribe: (fn: Listener) => () => void
}

const Ctx = createContext<WSContext>({ connected: false, subscribe: () => () => {} })

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false)
  const listenersRef = useRef<Set<Listener>>(new Set())
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let alive = true
    function connect() {
      if (!alive) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/events`)
      wsRef.current = ws
      ws.onopen = () => setConnected(true)
      ws.onclose = () => { setConnected(false); if (alive) setTimeout(connect, 3000) }
      ws.onerror = () => ws.close()
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          listenersRef.current.forEach((fn) => fn(data))
        } catch {}
      }
    }
    connect()
    return () => { alive = false; wsRef.current?.close() }
  }, [])

  const subscribe = useCallback((fn: Listener) => {
    listenersRef.current.add(fn)
    return () => { listenersRef.current.delete(fn) }
  }, [])

  return <Ctx.Provider value={{ connected, subscribe }}>{children}</Ctx.Provider>
}

export function useWebSocket() { return useContext(Ctx) }

import { useWebSocket } from '../hooks/useWebSocket'
import { Wifi, WifiOff } from 'lucide-react'

export function Statusbar() {
  const { connected } = useWebSocket()
  return (
    <footer className="h-7 bg-surface border-t border-border flex items-center px-4 text-[11px] text-muted gap-4 shrink-0 select-none">
      <span className="flex items-center gap-1.5">
        {connected ? (
          <>
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-50 animate-ping" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-success" />
            </span>
            <Wifi size={11} className="text-success/70" />
            <span className="text-muted-light">已连接</span>
          </>
        ) : (
          <>
            <span className="h-1.5 w-1.5 rounded-full bg-danger" />
            <WifiOff size={11} className="text-danger/70" />
            <span className="text-danger/80">已断开</span>
          </>
        )}
      </span>
      <span className="ml-auto text-muted/50 font-mono text-[10px]">Warehouse Vision v2.0</span>
    </footer>
  )
}

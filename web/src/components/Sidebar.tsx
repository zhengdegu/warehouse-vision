import { NavLink } from 'react-router-dom'
import { Video, Bell, Settings, Brain, Activity, Shield, BarChart3 } from 'lucide-react'

const nav = [
  { to: '/live', icon: Video, label: '监控' },
  { to: '/events', icon: Bell, label: '事件' },
  { to: '/accuracy', icon: BarChart3, label: '准确率' },
  { to: '/config', icon: Settings, label: '配置' },
  { to: '/training', icon: Brain, label: '训练' },
  { to: '/system', icon: Activity, label: '系统' },
]

export function Sidebar() {
  return (
    <nav className="w-16 bg-surface flex flex-col items-center py-5 gap-1.5 shrink-0 border-r border-border">
      {/* Logo */}
      <div className="w-9 h-9 rounded-xl bg-accent-dim flex items-center justify-center mb-5 animate-glow">
        <Shield size={18} className="text-accent" />
      </div>

      {nav.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            `relative flex flex-col items-center justify-center w-11 h-11 rounded-xl text-[10px] gap-0.5 cursor-pointer transition-all duration-200 ${
              isActive
                ? 'bg-accent-dim text-accent-light'
                : 'text-muted hover:text-muted-light hover:bg-card'
            }`
          }
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <span className="absolute -left-px top-1/2 -translate-y-1/2 w-[3px] h-4 rounded-r-full bg-accent" />
              )}
              <Icon size={17} strokeWidth={isActive ? 2.2 : 1.6} />
              <span className="font-medium leading-none">{label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  )
}

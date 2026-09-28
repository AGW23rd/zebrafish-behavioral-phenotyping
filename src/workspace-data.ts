export type TrackPoint = {
  time: number
  frame: number
  motion: number
  tracked?: boolean
  x_percent?: number | null
  y_percent?: number | null
  area_percent?: number | null
  orientation_deg?: number | null
  speed_px_s?: number | null
  turn_rate_deg_s?: number | null
}

export type AnalysisSummary = {
  mean_speed_px_s: number | null
  max_speed_px_s: number | null
  distance_px: number | null
  mean_area_percent: number | null
  edge_occupancy_percent: number | null
  center_occupancy_percent: number | null
  valid_speed_intervals: number
}

export type SavedSession = {
  id: string
  workspaceId?: string
  name: string
  fileName: string
  fileBytes: number
  createdAt: string
  uploadedAt?: string
  duration: number
  frameRate: number
  points: TrackPoint[]
  trackedPercent?: number
  sampleRateHz?: number
  summary?: AnalysisSummary
  pixelThreshold?: number
  activityThreshold?: number
  tracker?: string
  opencvVersion?: string
}

export type ResearchWorkspace = {
  id: string
  name: string
  createdAt: string
}

export type AppPreferences = {
  pixelThreshold: number
  activityThreshold: number
  frameRate: number
}

export const defaultPreferences: AppPreferences = {
  pixelThreshold: 10,
  activityThreshold: 0.1,
  frameRate: 0,
}

const SESSION_KEY = 'shoal.sessions.v1'
const PREFERENCES_KEY = 'shoal.preferences.v2'
const WORKSPACE_KEY = 'shoal.workspaces.v1'
const ACTIVE_WORKSPACE_KEY = 'shoal.active-workspace.v1'

const defaultWorkspace: ResearchWorkspace = {
  id: 'research-lab',
  name: 'Research lab',
  createdAt: new Date(0).toISOString(),
}

export function loadWorkspaces(): ResearchWorkspace[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(WORKSPACE_KEY) ?? '[]')
    const workspaces = Array.isArray(stored) ? stored.filter(isWorkspace) : []
    return workspaces.length ? workspaces : [defaultWorkspace]
  } catch {
    return [defaultWorkspace]
  }
}

export function storeWorkspaces(workspaces: ResearchWorkspace[]) {
  localStorage.setItem(WORKSPACE_KEY, JSON.stringify(workspaces))
}

export function loadActiveWorkspaceId(workspaces: ResearchWorkspace[]) {
  try {
    const stored = localStorage.getItem(ACTIVE_WORKSPACE_KEY)
    return workspaces.some((workspace) => workspace.id === stored) ? stored! : workspaces[0].id
  } catch {
    return workspaces[0].id
  }
}

export function storeActiveWorkspaceId(id: string) {
  localStorage.setItem(ACTIVE_WORKSPACE_KEY, id)
}

export function loadSessions(): SavedSession[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(SESSION_KEY) ?? '[]')
    return Array.isArray(stored) ? stored.filter(isSavedSession) : []
  } catch {
    return []
  }
}

export function storeSessions(sessions: SavedSession[]) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(sessions))
}

export function loadPreferences(): AppPreferences {
  try {
    const stored = JSON.parse(localStorage.getItem(PREFERENCES_KEY) ?? '{}') as Partial<AppPreferences>
    return {
      pixelThreshold: boundedNumber(stored.pixelThreshold, defaultPreferences.pixelThreshold, 1, 100),
      activityThreshold: boundedNumber(stored.activityThreshold, defaultPreferences.activityThreshold, 0.02, 8),
      frameRate: boundedNumber(stored.frameRate, 0, 0, 1000),
    }
  } catch {
    return defaultPreferences
  }
}

export function storePreferences(preferences: AppPreferences) {
  localStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences))
}

export function summarizePoints(points: TrackPoint[], activityThreshold: number) {
  const activeCount = points.reduce((count, point) => count + Number(point.motion > activityThreshold), 0)
  const bouts = points.reduce((count, point, index) => {
    return point.motion > activityThreshold && (index === 0 || points[index - 1].motion <= activityThreshold) ? count + 1 : count
  }, 0)
  const activeSeconds = activeCount * 0.5
  return {
    activeShare: points.length ? Math.round(activeCount / points.length * 100) : 0,
    bouts,
    activeSeconds,
    meanBoutSeconds: bouts ? activeSeconds / bouts : 0,
    meanMotion: points.length ? points.reduce((sum, point) => sum + point.motion, 0) / points.length : 0,
  }
}

function boundedNumber(value: unknown, fallback: number, minimum: number, maximum: number) {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, value))
    : fallback
}

function isSavedSession(value: unknown): value is SavedSession {
  if (!value || typeof value !== 'object') return false
  const session = value as Partial<SavedSession>
  return typeof session.id === 'string'
    && typeof session.name === 'string'
    && typeof session.fileName === 'string'
    && typeof session.createdAt === 'string'
    && Array.isArray(session.points)
}

function isWorkspace(value: unknown): value is ResearchWorkspace {
  if (!value || typeof value !== 'object') return false
  const workspace = value as Partial<ResearchWorkspace>
  return typeof workspace.id === 'string'
    && typeof workspace.name === 'string'
    && typeof workspace.createdAt === 'string'
}

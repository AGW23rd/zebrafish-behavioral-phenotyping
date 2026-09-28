import {
  ArrowUpRight,
  Activity,
  CalendarDays,
  CheckCircle2,
  Database,
  FileSpreadsheet,
  FlaskConical,
  HelpCircle,
  RotateCcw,
  Settings2,
  Trash2,
} from 'lucide-react'
import type { AppPreferences, SavedSession } from './workspace-data'

export function SessionsView({
  sessions,
  onOpen,
  onDelete,
}: {
  sessions: SavedSession[]
  onOpen: (session: SavedSession) => void
  onDelete: (id: string) => void
}) {
  return (
    <section className="tab-page">
      <div className="tab-page-heading">
        <div><div className="eyebrow"><span className="eyebrow-line" /> LOCAL LIBRARY</div><h2>Saved sessions</h2><p>Analysis results are stored in this browser. Original video files are never saved.</p></div>
        <span className="session-count">{sessions.length} {sessions.length === 1 ? 'session' : 'sessions'}</span>
      </div>
      {sessions.length ? (
        <div className="sessions-table-wrap">
          <table className="sessions-table">
            <thead><tr><th>SESSION</th><th>SOURCE</th><th>DATE SAVED</th><th>POINTS</th><th>TRACK COVERAGE</th><th>MEAN MOTION</th><th /></tr></thead>
            <tbody>{sessions.map((session) => <tr key={session.id}>
              <td><strong>{session.name}</strong></td>
              <td>{session.fileName}<small>{formatBytes(session.fileBytes)}</small></td>
              <td>{formatDate(session.createdAt)}</td>
              <td>{session.points.length.toLocaleString()}</td>
              <td>{session.trackedPercent == null ? '—' : `${session.trackedPercent.toFixed(1)}%`}</td>
              <td>{meanMotion(session.points).toFixed(1)}%</td>
              <td className="session-actions"><button className="table-action" onClick={() => onOpen(session)} title="Open saved results" aria-label={`Open ${session.name}`}><ArrowUpRight size={15} /></button><button className="table-action danger" onClick={() => onDelete(session.id)} title="Delete session" aria-label={`Delete ${session.name}`}><Trash2 size={15} /></button></td>
            </tr>)}</tbody>
          </table>
        </div>
      ) : (
        <div className="tab-empty"><span><CalendarDays size={20} /></span><h3>No saved sessions</h3><p>Run an analysis and choose Save session to keep its results in this browser.</p></div>
      )}
      <div className="storage-note"><Database size={15} /><span>Session data stays in local browser storage. Video files are not included.</span></div>
    </section>
  )
}

export function PreferencesView({
  preferences,
  onChange,
  onReset,
}: {
  preferences: AppPreferences
  onChange: (preferences: AppPreferences) => void
  onReset: () => void
}) {
  return (
    <section className="tab-page preferences-page">
      <div className="tab-page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> WORKSPACE</div><h2>Preferences</h2><p>Defaults used when setting up the next recording analysis.</p></div><button className="reset-button" onClick={onReset}><RotateCcw size={14} /> Reset defaults</button></div>
      <div className="preferences-list">
        <PreferenceRow title="Pixel-change sensitivity" description="Lower values respond to smaller changes between frames." value={`${preferences.pixelThreshold}`}>
          <input type="range" min="1" max="100" value={preferences.pixelThreshold} onChange={(event) => onChange({ ...preferences, pixelThreshold: Number(event.target.value) })} aria-label="Default pixel-change sensitivity" />
        </PreferenceRow>
        <PreferenceRow title="Active signal threshold" description="A sample above this changed-pixel share counts as active." value={`${preferences.activityThreshold.toFixed(1)}%`}>
          <input type="range" min="0.02" max="8" step="0.02" value={preferences.activityThreshold} onChange={(event) => onChange({ ...preferences, activityThreshold: Number(event.target.value) })} aria-label="Default active signal threshold" />
        </PreferenceRow>
        <PreferenceRow title="Default source frame rate" description="Leave at automatic to use the video metadata when available." value={preferences.frameRate > 0 ? `${preferences.frameRate} fps` : 'Automatic'}>
          <div className="input-with-unit preference-number"><input type="number" min="0" max="1000" step="1" value={preferences.frameRate || ''} placeholder="Automatic" onChange={(event) => onChange({ ...preferences, frameRate: Math.max(0, Number(event.target.value)) })} aria-label="Default source frame rate" /><span>fps</span></div>
        </PreferenceRow>
      </div>
      <div className="privacy-setting"><div><strong>Local processing</strong><p>Video decoding and tracking use the OpenCV service at 127.0.0.1. Recordings are not sent to a remote server.</p></div><span className="local-badge"><span /> On this device</span></div>
    </section>
  )
}

export function HelpView({ onNavigate }: { onNavigate: (tab: 'workspace' | 'analysis' | 'sessions' | 'preferences') => void }) {
  return (
    <section className="tab-page help-page">
      <div className="tab-page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> METHODS & WORKFLOW</div><h2>Help & analysis guide</h2><p>How Shoal Lab processes a video and how to interpret its measurements.</p></div><HelpCircle size={24} color="#5c8068" /></div>
      <div className="help-grid">
        <article className="help-section"><h3><FileSpreadsheet size={16} /> Start an analysis</h3><ol><li>Choose a clear side-view video with a stationary camera and steady illumination.</li><li>Use the source frame rate reported by the file; set an override in Preferences only if needed.</li><li>Adjust pixel sensitivity and the active-signal threshold to match image noise.</li><li>Run the local analysis, inspect the extracted frame, trajectory coverage, and missed frames, then export the Excel report.</li></ol><button className="help-action" onClick={() => onNavigate('analysis')}>Open behavior analysis <ArrowUpRight size={14} /></button></article>
        <article className="help-section"><h3><Activity size={16} /> Measurements</h3><dl><dt>Track coverage</dt><dd>Share of sampled frames where the selected moving foreground component was detected and associated to the previous centroid.</dd><dt>Speed and distance</dt><dd>Centroid displacement is reported in pixels and pixels per second. These are not physical units without spatial calibration.</dd><dt>Area and orientation</dt><dd>Foreground contour area and image-plane principal-axis orientation; orientation is ambiguous by 180 degrees.</dd><dt>Edge/center occupancy</dt><dd>Fraction of detected positions within the outer 15% of the image or interior.</dd><dt>Movement bouts</dt><dd>Runs of sampled global pixel-change values above the configured activity threshold.</dd></dl></article>
        <article className="help-section"><h3><CheckCircle2 size={16} /> Research use</h3><p>This is classical OpenCV foreground segmentation and centroid association, not deep-learning pose estimation or a validated fish detector. It follows one moving foreground component and can select reflections, bubbles, debris, or shadows instead of a fish. Inspect the extracted track, preserve source videos and annotation protocols, and compare against manually labelled examples before drawing scientific conclusions.</p><p>Tail-beat frequency, midline curvature, multi-fish identity, and validated species recognition are not measured. Research validity depends on validation for your camera, species, and protocol.</p></article>
        <article className="help-section"><h3><Database size={16} /> Files & privacy</h3><p>All analysis runs on this computer. Video files are streamed to the loopback-only OpenCV service, temporarily written to disk, and removed when processing finishes. Saved sessions contain measurements and file metadata, not video bytes.</p><button className="help-action" onClick={() => onNavigate('sessions')}>View saved sessions <ArrowUpRight size={14} /></button></article>
      </div>
    </section>
  )
}

export function ResearchEnvironmentView({
  workspaceName,
  sessionCount,
  onNavigate,
}: {
  workspaceName: string
  sessionCount: number
  onNavigate: (tab: 'workspace' | 'analysis' | 'sessions' | 'preferences') => void
}) {
  return (
    <section className="tab-page profile-page">
      <div className="tab-page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> LOCAL RESEARCH ENVIRONMENT</div><h2>Research environment</h2><p>Project context and data-handling status for this browser.</p></div><span className="profile-badge">RE</span></div>
      <div className="environment-list">
        <div><span className="environment-icon"><FlaskConical size={17} /></span><span><strong>Active workspace</strong><small>{workspaceName}</small></span><button onClick={() => onNavigate('workspace')}>Manage</button></div>
        <div><span className="environment-icon"><Database size={17} /></span><span><strong>Saved analyses</strong><small>{sessionCount} result sets in this workspace</small></span><button onClick={() => onNavigate('sessions')}>Open sessions</button></div>
        <div><span className="environment-icon"><CheckCircle2 size={17} /></span><span><strong>Processing location</strong><small>Local computer · loopback-only API · no remote upload</small></span><span className="environment-state">LOCAL</span></div>
        <div><span className="environment-icon"><Settings2 size={17} /></span><span><strong>Analysis defaults</strong><small>Pixel threshold, activity cutoff, and source frame rate</small></span><button onClick={() => onNavigate('preferences')}>Preferences</button></div>
      </div>
      <div className="environment-note"><strong>Designed by Andrew G. Watts</strong><span>This environment records analysis settings and outputs locally. Use your institution’s approved storage and validation practices for study data.</span></div>
    </section>
  )
}

function PreferenceRow({ title, description, value, children }: { title: string; description: string; value: string; children: React.ReactNode }) {
  return <div className="preference-row"><div className="preference-copy"><strong>{title}</strong><p>{description}</p></div><div className="preference-control">{children}<span>{value}</span></div></div>
}

function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Unknown' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—'
  return bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GB`
    : bytes >= 1024 ** 2
      ? `${(bytes / 1024 ** 2).toFixed(1)} MB`
      : `${Math.max(1, Math.round(bytes / 1024))} KB`
}

function meanMotion(points: SavedSession['points']) {
  return points.length ? points.reduce((sum, point) => sum + point.motion, 0) / points.length : 0
}

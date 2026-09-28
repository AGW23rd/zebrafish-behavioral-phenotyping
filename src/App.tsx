import { useMemo, useRef, useState } from 'react'
import type { ChangeEvent, ReactNode, SyntheticEvent } from 'react'
import {
  Activity,
  ArrowDownToLine,
  BookmarkPlus,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  FileVideo,
  FileSpreadsheet,
  FlaskConical,
  Fish,
  Focus,
  Info,
  Maximize2,
  Play,
  ScanLine,
  Settings2,
  Upload,
  Waves,
} from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'
import { HelpView, PreferencesView, ResearchEnvironmentView, SessionsView } from './WorkspaceTabs'
import { WorkspaceView } from './WorkspaceView'
import type { AnalysisSummary, SavedSession, TrackPoint } from './workspace-data'
import type { ResearchWorkspace } from './workspace-data'
import { defaultPreferences, loadActiveWorkspaceId, loadPreferences, loadSessions, loadWorkspaces, storeActiveWorkspaceId, storePreferences, storeSessions, storeWorkspaces } from './workspace-data'
import { buildXlsxWorkbook } from './xlsx-export'

const VIDEO_ACCEPT = 'video/*,.avi,.mkv,.mpg,.mpeg,.m4v'

function App() {
  const initialPreferences = loadPreferences()
  const [activeTab, setActiveTab] = useState<'workspace' | 'analysis' | 'sessions' | 'preferences' | 'help' | 'profile'>('analysis')
  const [workspaces, setWorkspaces] = useState<ResearchWorkspace[]>(loadWorkspaces)
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(() => loadActiveWorkspaceId(loadWorkspaces()))
  const [videoUrl, setVideoUrl] = useState('')
  const [hasSelectedFile, setHasSelectedFile] = useState(false)
  const [usesLocalDecoder, setUsesLocalDecoder] = useState(false)
  const [aviPreview, setAviPreview] = useState('')
  const [videoName, setVideoName] = useState('')
  const [videoBytes, setVideoBytes] = useState(0)
  const [uploadedAt, setUploadedAt] = useState('')
  const [trackPoints, setTrackPoints] = useState<TrackPoint[]>([])
  const [analysisSummary, setAnalysisSummary] = useState<AnalysisSummary | null>(null)
  const [trackedPercent, setTrackedPercent] = useState(0)
  const [sampleRateHz, setSampleRateHz] = useState(5)
  const [trackingStatus, setTrackingStatus] = useState<'tracked' | 'low_coverage' | 'not_detected'>('not_detected')
  const [trackerName, setTrackerName] = useState('MOG2 foreground segmentation + centroid association')
  const [opencvVersion, setOpencvVersion] = useState('')
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>(loadSessions)
  const [isTracking, setIsTracking] = useState(false)
  const [threshold, setThreshold] = useState(initialPreferences.pixelThreshold)
  const [activityThreshold, setActivityThreshold] = useState(initialPreferences.activityThreshold)
  const [frameRate, setFrameRate] = useState(initialPreferences.frameRate)
  const [sessionName, setSessionName] = useState('Untitled session')
  const [notice, setNotice] = useState('')
  const [, setIsPlaying] = useState(false)
  const [uploadPhase, setUploadPhase] = useState<'idle' | 'uploading' | 'decoding'>('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  const selectedFileRef = useRef<File | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const previousFrameRef = useRef<Uint8ClampedArray | null>(null)
  const lastSampleTimeRef = useRef(0)
  const trackPointsRef = useRef<TrackPoint[]>([])
  const lastTimelineSecondRef = useRef(-1)
  const activeWorkspace = workspaces.find((workspace) => workspace.id === activeWorkspaceId) ?? workspaces[0]
  const workspaceSessions = savedSessions.filter((session) => (session.workspaceId ?? 'research-lab') === activeWorkspaceId)

  const updatePreferences = (next: { pixelThreshold: number; activityThreshold: number; frameRate: number }) => {
    setThreshold(next.pixelThreshold)
    setActivityThreshold(next.activityThreshold)
    setFrameRate(next.frameRate)
    try {
      storePreferences(next)
    } catch {
      setNotice('Browser storage is unavailable; preferences will not persist after reload.')
    }
  }

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    if (!file) return
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    const nextUrl = ''
    selectedFileRef.current = file
    setHasSelectedFile(true)
    setVideoUrl(nextUrl)
    setUsesLocalDecoder(true)
    setAviPreview('')
    setAnalysisSummary(null)
    setTrackedPercent(0)
    setTrackingStatus('not_detected')
    setOpencvVersion('')
    setVideoName(file.name)
    setVideoBytes(file.size)
    setUploadedAt(new Date().toISOString())
    setSessionName(file.name.replace(/\.[^.]+$/, ''))
    setTrackPoints([])
    trackPointsRef.current = []
    previousFrameRef.current = null
    setIsTracking(false)
    setUploadPhase('idle')
    setUploadProgress(0)
    setVideoTime(0)
    setVideoDuration(0)
    setNotice('Video selected for local OpenCV tracking. Source frame rate is read from the file when available.')
    event.currentTarget.value = ''
  }

  const handleMetadata = (event: SyntheticEvent<HTMLVideoElement>) => {
    const video = event.currentTarget
    setVideoDuration(video.duration)
    setNotice(`${video.videoWidth} × ${video.videoHeight} · ${formatTime(video.duration)}`)
  }

  const sampleMotion = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!isTracking || !video || !canvas || video.paused || video.ended || !video.videoWidth) return
    if (video.currentTime - lastSampleTimeRef.current < 0.5) return

    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) return
    const scale = Math.min(1, 180 / video.videoWidth, 180 / video.videoHeight)
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data
    const previous = previousFrameRef.current
    previousFrameRef.current = new Uint8ClampedArray(pixels)
    lastSampleTimeRef.current = video.currentTime
    if (!previous) return

    let changed = 0
    for (let y = 2; y < canvas.height - 2; y += 2) {
      for (let x = 2; x < canvas.width - 2; x += 2) {
        const index = (y * canvas.width + x) * 4
          const luminanceDelta = Math.abs(
            0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2]
            - 0.2126 * previous[index] - 0.7152 * previous[index + 1] - 0.0722 * previous[index + 2],
          )
        if (luminanceDelta > threshold) {
          changed += 1
        }
      }
    }

    const point: TrackPoint = {
      time: video.currentTime,
      frame: Math.round(video.currentTime * frameRate),
      motion: Math.round((changed / (((canvas.width - 4) / 2) * ((canvas.height - 4) / 2))) * 1000) / 10,
    }
    trackPointsRef.current.push(point)
    if (trackPointsRef.current.length % 8 === 0) setTrackPoints([...trackPointsRef.current])
  }

  const handleVideoTimeUpdate = () => {
    const currentTime = videoRef.current?.currentTime ?? 0
    if (Math.floor(currentTime) !== lastTimelineSecondRef.current) {
      lastTimelineSecondRef.current = Math.floor(currentTime)
      setVideoTime(currentTime)
    }
    sampleMotion()
  }

  const handleVideoError = () => {
    const file = selectedFileRef.current
    if (file && !usesLocalDecoder) {
      setUsesLocalDecoder(true)
      analyzeAvi(file)
      return
    }
    setIsTracking(false)
    setNotice('The local decoder could not read this video. Check the file and its installed codec support.')
  }

  const analyzeAvi = (file: File) => {
    const query = new URLSearchParams({
      filename: file.name,
      frame_rate: String(frameRate),
      pixel_threshold: String(threshold),
    })
    const request = new XMLHttpRequest()
    setIsTracking(true)
    setUploadPhase('uploading')
    setUploadProgress(0)
    setNotice('Sending the AVI to the local decoder. The file stays on this device.')
    request.open('POST', `/api/analyze?${query.toString()}`)
    request.setRequestHeader('Content-Type', 'application/octet-stream')
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) setUploadProgress(Math.round(event.loaded / event.total * 100))
    }
    request.upload.onload = () => {
      setUploadPhase('decoding')
      setNotice('Upload complete. Decoding and sampling the video locally...')
    }
    request.onerror = () => {
      setIsTracking(false)
      setUploadPhase('idle')
      setNotice('Could not reach the local decoder. Start both services with npm run dev and try again.')
    }
    request.onload = () => {
      setIsTracking(false)
      setUploadPhase('idle')
      let result: { detail?: string; width?: number; height?: number; frame_rate?: number; duration?: number; preview_jpeg?: string; points?: TrackPoint[]; summary?: AnalysisSummary; tracked_percent?: number; sample_rate_hz?: number; tracking_status?: 'tracked' | 'low_coverage' | 'not_detected'; tracker?: string; opencv_version?: string }
      try {
        result = JSON.parse(request.responseText)
      } catch {
        setNotice('The local decoder returned an unreadable response.')
        return
      }
      if (request.status < 200 || request.status >= 300) {
        setNotice(result.detail || `Local analysis failed (${request.status}).`)
        return
      }
      if (!result.points?.length || !result.frame_rate || !result.duration) {
        setNotice('The local decoder returned no usable analysis samples.')
        return
      }
      setTrackPoints(result.points)
      trackPointsRef.current = result.points
      setAnalysisSummary(result.summary ?? null)
      setTrackedPercent(result.tracked_percent ?? 0)
      setSampleRateHz(result.sample_rate_hz ?? 5)
      setTrackingStatus(result.tracking_status ?? 'not_detected')
      setTrackerName(result.tracker ?? 'MOG2 foreground segmentation + centroid association')
      setOpencvVersion(result.opencv_version ?? '')
      setVideoDuration(result.duration)
      setVideoTime(result.duration)
      setFrameRate(frameRate || result.frame_rate)
      setAviPreview(result.preview_jpeg ? `data:image/jpeg;base64,${result.preview_jpeg}` : '')
      setNotice(`${result.width} × ${result.height} · ${formatTime(result.duration)} · ${result.points.length} samples · ${result.tracked_percent ?? 0}% track coverage`)
    }
    request.send(file)
  }

  const startTracking = () => {
    const file = selectedFileRef.current
    if (!file) {
      setNotice('Upload a video before starting analysis.')
      return
    }
    if (usesLocalDecoder) {
      analyzeAvi(file)
      return
    }
    if (frameRate < 1) {
      setNotice('Enter the recording frame rate before starting analysis.')
      return
    }
    previousFrameRef.current = null
    lastSampleTimeRef.current = 0
    lastTimelineSecondRef.current = -1
    setTrackPoints([])
    trackPointsRef.current = []
    setIsTracking(true)
    setNotice('Analyzing movement signal during playback. You can pause and resume the recording.')
    const video = videoRef.current
    if (!video) return
    if (video.currentTime > 0.05) video.currentTime = 0
    void video.play().catch((error: unknown) => {
      setIsTracking(false)
      const blocked = error instanceof Error && error.name === 'NotAllowedError'
      setNotice(blocked
        ? 'Playback was blocked by the browser. Start playback from the video controls to continue.'
        : error instanceof Error && error.name === 'AbortError'
          ? 'Video playback was interrupted. Press play on the video controls to resume analysis.'
          : 'The browser could not play this recording. Try an H.264 MP4 video.')
    })
  }

  const exportCsv = () => {
    const points = trackPointsRef.current.length ? trackPointsRef.current : trackPoints
    if (!points.length) return
    const rows = ['time_s,frame,tracked,x_percent,y_percent,area_percent,orientation_deg,speed_px_s,turn_rate_deg_s,pixel_change_percent,active', ...points.map((point) =>
      [point.time.toFixed(3), point.frame, point.tracked ? 1 : 0, point.x_percent ?? '', point.y_percent ?? '', point.area_percent ?? '', point.orientation_deg ?? '', point.speed_px_s ?? '', point.turn_rate_deg_s ?? '', point.motion, point.motion > activityThreshold ? 1 : 0].join(','),
    )]
    const file = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(file)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${sessionName || 'fish-track'}-track.csv`
    anchor.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const exportExcel = () => {
    const points = trackPointsRef.current.length ? trackPointsRef.current : trackPoints
    if (!points.length) return
    const summaryRows: (string | number | null)[][] = [
      ['Session', sessionName],
      ['Source video', videoName],
      ['Video size (bytes)', videoBytes],
      ['Video selected (UTC)', uploadedAt],
      ['Duration (s)', videoDuration],
      ['Source frame rate (fps)', frameRate || 'Read from video'],
      ['Tracker', trackerName],
      ['OpenCV version', opencvVersion],
      ['Analysis sample rate (Hz)', sampleRateHz],
      ['Tracking coverage (%)', trackedPercent],
      ['Track status', trackingStatus],
      ['Mean movement signal (%)', meanMotion],
      ['Active samples (%)', activeShare],
      ['Movement bouts', bouts],
      ['Mean bout duration (s)', meanBoutSeconds],
      ['Mean tracked speed (px/s)', analysisSummary?.mean_speed_px_s ?? null],
      ['Maximum tracked speed (px/s)', analysisSummary?.max_speed_px_s ?? null],
      ['Track distance (px)', analysisSummary?.distance_px ?? null],
      ['Mean detected area (%)', analysisSummary?.mean_area_percent ?? null],
      ['Arena edge occupancy (%)', analysisSummary?.edge_occupancy_percent ?? null],
      ['Arena center occupancy (%)', analysisSummary?.center_occupancy_percent ?? null],
      ['Valid speed intervals', analysisSummary?.valid_speed_intervals ?? 0],
      ['Motion pixel threshold', threshold],
      ['Active signal threshold (%)', activityThreshold],
      ['Coordinate units', 'Percent of frame; not calibrated distance'],
    ]
    const pointRows = points.map((point) => [
      point.time, point.frame, point.tracked ? 'yes' : 'no', point.x_percent ?? null,
      point.y_percent ?? null, point.area_percent ?? null, point.orientation_deg ?? null,
      point.speed_px_s ?? null, point.turn_rate_deg_s ?? null, point.motion,
      point.motion > activityThreshold ? 'active' : 'inactive',
    ])
    const workbook = buildXlsxWorkbook([
      { name: 'Movement Summary', headers: ['Measure', 'Value'], rows: summaryRows },
      { name: 'Track Samples', headers: ['Time (s)', 'Frame', 'Tracked', 'X (%)', 'Y (%)', 'Area (%)', 'Orientation (deg)', 'Speed (px/s)', 'Turn rate (deg/s)', 'Pixel change (%)', 'Activity'], rows: pointRows },
    ])
    const url = URL.createObjectURL(new Blob([new Uint8Array(workbook)], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${sessionName || 'fish-analysis'}-movement-summary.xlsx`
    anchor.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const saveCurrentSession = () => {
    const points = trackPointsRef.current.length ? trackPointsRef.current : trackPoints
    if (!points.length) return
    const session: SavedSession = {
      id: crypto.randomUUID(),
      workspaceId: activeWorkspaceId,
      name: sessionName.trim() || videoName || 'Untitled session',
      fileName: videoName || 'Unknown video',
      fileBytes: videoBytes,
      createdAt: new Date().toISOString(),
      uploadedAt,
      duration: videoDuration,
      frameRate,
      points,
      trackedPercent,
      sampleRateHz,
      summary: analysisSummary ?? undefined,
      pixelThreshold: threshold,
      activityThreshold,
      tracker: trackerName,
      opencvVersion,
    }
    const nextSessions = [session, ...savedSessions].slice(0, 20)
    try {
      storeSessions(nextSessions)
      setSavedSessions(nextSessions)
      setNotice('Session results saved in this browser. The video itself was not saved.')
    } catch {
      setNotice('Could not save this session. Browser storage may be full; export the CSV instead.')
    }
  }

  const openSavedSession = (session: SavedSession) => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    selectedFileRef.current = null
    setHasSelectedFile(false)
    setVideoUrl('')
    setUsesLocalDecoder(false)
    setAviPreview('')
    setVideoName(session.fileName)
    setVideoBytes(session.fileBytes)
    setUploadedAt(session.uploadedAt ?? session.createdAt)
    setSessionName(session.name)
    setVideoDuration(session.duration)
    setVideoTime(session.duration)
    setFrameRate(session.frameRate)
    setThreshold(session.pixelThreshold ?? defaultPreferences.pixelThreshold)
    setActivityThreshold(session.activityThreshold ?? defaultPreferences.activityThreshold)
    setTrackerName(session.tracker ?? 'MOG2 foreground segmentation + centroid association')
    setOpencvVersion(session.opencvVersion ?? '')
    setAnalysisSummary(session.summary ?? null)
    const savedCoverage = session.trackedPercent ?? (session.points.filter((point) => point.tracked).length / Math.max(session.points.length, 1)) * 100
    setTrackedPercent(savedCoverage)
    setSampleRateHz(session.sampleRateHz ?? 5)
    setTrackingStatus(savedCoverage >= 60 ? 'tracked' : savedCoverage > 0 ? 'low_coverage' : 'not_detected')
    if (session.workspaceId && session.workspaceId !== activeWorkspaceId) {
      setActiveWorkspaceId(session.workspaceId)
      storeActiveWorkspaceId(session.workspaceId)
    }
    setTrackPoints(session.points)
    trackPointsRef.current = session.points
    setIsTracking(false)
    setActiveTab('analysis')
    setNotice('Saved results loaded. The original video is not stored in sessions.')
  }

  const deleteSavedSession = (id: string) => {
    const nextSessions = savedSessions.filter((session) => session.id !== id)
    try {
      storeSessions(nextSessions)
      setSavedSessions(nextSessions)
    } catch {
      setNotice('Could not update the saved sessions in browser storage.')
    }
  }

  const resetPreferences = () => {
    updatePreferences(defaultPreferences)
  }

  const selectWorkspace = (id: string) => {
    if (!workspaces.some((workspace) => workspace.id === id)) return
    setActiveWorkspaceId(id)
    try {
      storeActiveWorkspaceId(id)
    } catch {
      setNotice('Browser storage is unavailable; the workspace selection will not persist after reload.')
    }
  }

  const createWorkspace = (name: string) => {
    const workspace = { id: crypto.randomUUID(), name, createdAt: new Date().toISOString() }
    const nextWorkspaces = [...workspaces, workspace]
    try {
      storeWorkspaces(nextWorkspaces)
      setWorkspaces(nextWorkspaces)
      setActiveWorkspaceId(workspace.id)
      storeActiveWorkspaceId(workspace.id)
      setNotice(`Workspace “${name}” created.`)
    } catch {
      setNotice('Could not save the new workspace in browser storage.')
    }
  }

  const deleteWorkspace = (id: string) => {
    if (workspaces.length <= 1) return
    const nextWorkspaces = workspaces.filter((workspace) => workspace.id !== id)
    const nextSessions = savedSessions.filter((session) => (session.workspaceId ?? 'research-lab') !== id)
    try {
      storeWorkspaces(nextWorkspaces)
      storeSessions(nextSessions)
      setWorkspaces(nextWorkspaces)
      setSavedSessions(nextSessions)
      if (activeWorkspaceId === id) {
        setActiveWorkspaceId(nextWorkspaces[0].id)
        storeActiveWorkspaceId(nextWorkspaces[0].id)
      }
    } catch {
      setNotice('Could not remove this workspace from browser storage.')
    }
  }
    const toggleFullscreen = () => {
      const stage = document.querySelector('.video-stage')
      if (document.fullscreenElement) {
        void document.exitFullscreen()
      } else if (stage?.requestFullscreen) {
        void stage.requestFullscreen().catch(() => setNotice('Fullscreen is unavailable in this browser.'))
      }
    }

  const sampleSeconds = 1 / Math.max(sampleRateHz, 0.1)
  const analysisStats = useMemo(() => {
    const activeCount = trackPoints.reduce((count, point) => count + Number(point.motion > activityThreshold), 0)
    const bouts = trackPoints.reduce((count, point, index) => {
      return point.motion > activityThreshold && (index === 0 || trackPoints[index - 1].motion <= activityThreshold) ? count + 1 : count
    }, 0)
    const activeSeconds = activeCount * sampleSeconds
    return {
      activeShare: trackPoints.length ? Math.round(activeCount / trackPoints.length * 100) : 0,
      bouts,
      activeSeconds,
      meanBoutSeconds: bouts ? activeSeconds / bouts : 0,
      meanMotion: trackPoints.length ? trackPoints.reduce((sum, point) => sum + point.motion, 0) / trackPoints.length : 0,
    }
  }, [trackPoints, activityThreshold, sampleSeconds])
  const chartPoints = useMemo(() => {
    if (trackPoints.length <= 1200) return trackPoints
    const step = Math.ceil(trackPoints.length / 1200)
    return trackPoints.filter((_, index) => index % step === 0 || index === trackPoints.length - 1)
  }, [trackPoints])
  const { activeShare, bouts, activeSeconds, meanBoutSeconds, meanMotion } = analysisStats

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark"><Waves size={19} /></span><span>shoal<span className="brand-dot">.</span></span></div>
        <div className="workspace-label">WORKSPACE</div>
        <button className="workspace-switch" onClick={() => setActiveTab('workspace')} title="Open workspace manager"><span className="workspace-avatar">{activeWorkspace.name.slice(0, 1).toUpperCase()}</span><span>{activeWorkspace.name}<small>Local workspace</small></span><ChevronDown size={15} /></button>
        <button className={`nav-item ${activeTab === 'workspace' ? 'active' : ''}`} onClick={() => setActiveTab('workspace')}><FlaskConical size={17} /><span>Workspaces</span><span className="nav-count">{workspaces.length}</span></button>
        <div className="sidebar-section-label">ANALYSIS</div>
        <button className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}><ScanLine size={17} /><span>Behavior analysis</span>{activeTab === 'analysis' && <span className="nav-count">01</span>}</button>
        <button className={`nav-item ${activeTab === 'sessions' ? 'active' : ''}`} onClick={() => setActiveTab('sessions')}><Activity size={17} /><span>Sessions</span><span className="nav-count">{workspaceSessions.length}</span></button>
        <button className={`nav-item ${activeTab === 'preferences' ? 'active' : ''}`} onClick={() => setActiveTab('preferences')}><Settings2 size={17} /><span>Preferences</span></button>
        <div className="sidebar-bottom"><div className="privacy-mark"><span /><div><strong>Private by design</strong><small>Videos stay on this device</small></div></div><div className="sidebar-version">SHOAL LAB <span>PREVIEW 0.1</span></div></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="breadcrumbs"><button onClick={() => setActiveTab('workspace')}>Workspace</button><span>/</span> {activeTab === 'analysis' ? 'Behavior analysis' : activeTab === 'sessions' ? 'Sessions' : activeTab === 'preferences' ? 'Preferences' : activeTab === 'help' ? 'Help' : activeTab === 'profile' ? 'Research environment' : 'Workspace manager'} <span>/</span> <strong>{activeTab === 'analysis' ? (videoName || 'New analysis') : activeTab === 'sessions' ? activeWorkspace.name : activeTab === 'preferences' ? 'Defaults' : activeTab === 'help' ? 'Methods & workflow' : activeWorkspace.name}</strong></div><div className="topbar-actions"><span className="local-badge"><span /> Local only</span><button className="icon-button" title="Help & methods" aria-label="Help & methods" onClick={() => setActiveTab('help')}><CircleHelp size={18} /></button><button className="avatar-button" title="Research environment" aria-label="Research environment" onClick={() => setActiveTab('profile')}>RE</button></div></header>

        {activeTab === 'analysis' && <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> FISH BEHAVIOR STUDIO</div><h1>Behavior analysis</h1><p>Track foreground motion, swimming kinematics, and arena use.</p></div><div className={`heading-status status-${trackingStatus}`}><span className="status-light" /> {uploadPhase === 'uploading' ? `Uploading ${uploadProgress}%` : uploadPhase === 'decoding' ? 'Analyzing locally' : trackPoints.length ? `${trackedPercent.toFixed(0)}% track coverage` : 'Ready for recording'}</div></section>}

        {activeTab === 'analysis' && <div className="analysis-grid">
          <section className="video-column">
            <div className="section-toolbar"><div className="section-title"><h2>Recording</h2><span className="experimental-tag">LOCAL TRACK · {sampleRateHz} HZ</span></div><button className="subtle-button" title="Toggle fullscreen" aria-label="Toggle fullscreen" onClick={toggleFullscreen}><Maximize2 size={15} /></button></div>
            <div className={`video-stage ${videoUrl || aviPreview ? 'has-video' : ''}`}>
              {usesLocalDecoder && aviPreview ? <img className="avi-preview" src={aviPreview} alt="First frame extracted from the recording" /> : usesLocalDecoder ? <div className="avi-ready"><FileVideo size={34} /><strong>Ready for local tracking</strong><span>OpenCV analyzes the recording and extracts one moving foreground track.</span></div> : videoUrl ? <video ref={videoRef} src={videoUrl} onLoadedMetadata={handleMetadata} onTimeUpdate={handleVideoTimeUpdate} onPlay={() => setIsPlaying(true)} onPause={() => { setIsPlaying(false); setTrackPoints([...trackPointsRef.current]) }} onError={handleVideoError} onEnded={() => { setIsTracking(false); setIsPlaying(false); setTrackPoints([...trackPointsRef.current]); setNotice('Analysis complete. Export the sampled behavior data as CSV.') }} controls playsInline /> : <div className="empty-video"><div className="tank-art"><div className="tank-grain" /><div className="tank-fish tank-fish-one"><Fish size={52} strokeWidth={1.1} /></div><div className="tank-fish tank-fish-two"><Fish size={29} strokeWidth={1.1} /></div><span className="tank-orbit orbit-one" /><span className="tank-orbit orbit-two" /><span className="tank-label">01 / ARENA</span><span className="tank-depth">WATER COLUMN</span></div><div className="empty-video-copy"><strong>{trackPoints.length ? 'Saved analysis results' : 'Bring a recording into the arena'}</strong><span>{trackPoints.length ? 'The source video is not stored with this session.' : 'Choose an MP4, WebM, MOV, or AVI video from this device.'}</span></div><label className="upload-button"><Upload size={16} /> Choose video<input type="file" accept={VIDEO_ACCEPT} onChange={handleUpload} /></label><span className="local-hint"><Info size={13} /> Files stay on this device. Large videos are streamed to the local decoder.</span></div>}
              <canvas ref={canvasRef} className="sampling-canvas" aria-hidden="true" />
              {uploadPhase !== 'idle' && <div className="analysis-overlay"><span>{uploadPhase === 'uploading' ? `UPLOADING ${uploadProgress}%` : 'DECODING LOCALLY'}</span><strong>{uploadPhase === 'uploading' ? `${uploadProgress}%` : '...'}</strong><i className={uploadPhase === 'decoding' ? 'indeterminate' : ''} style={uploadPhase === 'uploading' ? { width: `${uploadProgress}%` } : undefined} /></div>}
              {isTracking && uploadPhase === 'decoding' && <div className="analysis-overlay"><span>TRACKING OBJECT</span><strong>{trackedPercent.toFixed(0)}%</strong><i className="indeterminate" /></div>}
            </div>
            <div className="video-meta"><div className="recording-name"><span className="file-icon"><Fish size={15} /></span><span><strong>{videoName || 'No recording selected'}</strong><small>{videoName ? `${formatBytes(videoBytes)} · ${notice}` : 'MP4 · WebM · MOV · AVI'}</small></span></div><label className="replace-link">{videoName ? 'Replace video' : 'Browse files'}<input type="file" accept={VIDEO_ACCEPT} onChange={handleUpload} /></label></div>

            <div className="timeline-panel"><div className="timeline-heading"><span><Clock3 size={15} /> TIMELINE</span><span>{videoDuration ? formatTime(videoTime) : '00:00'} <i>/</i> {videoDuration ? formatTime(videoDuration) : '00:00'}</span></div><div className="timeline-track"><div className="timeline-progress" style={{ width: videoDuration ? `${(videoTime / videoDuration) * 100}%` : '0%' }} /><button className="timeline-knob" style={{ left: videoDuration ? `${(videoTime / videoDuration) * 100}%` : '0%' }} title="Current video position" /></div><div className="timeline-labels"><span>00:00</span><span>{videoDuration ? formatTime(videoDuration / 2) : '00:00'}</span><span>{videoDuration ? formatTime(videoDuration) : '00:00'}</span></div></div>

            <div className="results-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> SESSION READOUT · {trackPoints.length ? trackingStatus.replace('_', ' ').toUpperCase() : 'NO ANALYSIS'}</div><h2>Movement summary</h2></div><div className="results-actions"><button className="save-button" onClick={saveCurrentSession} disabled={!trackPoints.length}><BookmarkPlus size={15} /> Save session</button><button className="export-button" onClick={exportCsv} disabled={!trackPoints.length}><ArrowDownToLine size={15} /> CSV</button><button className="export-button" onClick={exportExcel} disabled={!trackPoints.length}><FileSpreadsheet size={15} /> Excel report</button></div></div>
            <div className="metric-grid">
              <Metric label="TRACK COVERAGE" value={trackPoints.length ? trackedPercent.toFixed(1) : '—'} unit="%" icon={<Focus size={16} />} note={`${trackPoints.filter((point) => point.tracked).length} of ${trackPoints.length} samples`} />
              <Metric label="MEAN SPEED" value={analysisSummary?.mean_speed_px_s != null ? analysisSummary.mean_speed_px_s.toFixed(1) : '—'} unit="px/s" icon={<Waves size={16} />} note="Tracked centroid; image scale" />
              <Metric label="DISTANCE" value={analysisSummary?.distance_px != null ? analysisSummary.distance_px.toFixed(0) : '—'} unit="px" icon={<Activity size={16} />} note="Accumulated centroid path" />
              <Metric label="EDGE OCCUPANCY" value={analysisSummary?.edge_occupancy_percent != null ? analysisSummary.edge_occupancy_percent.toFixed(1) : '—'} unit="%" icon={<Focus size={16} />} note="Outer 15% of frame" />
              <Metric label="MEAN MOTION" value={trackPoints.length ? meanMotion.toFixed(3) : '—'} unit="%" icon={<Activity size={16} />} note="Frame-change signal" />
              <Metric label="ACTIVE SAMPLES" value={trackPoints.length ? `${activeShare}` : '—'} unit="%" icon={<Play size={15} />} note={`Still samples: ${trackPoints.length ? 100 - activeShare : '—'}%`} />
              <Metric label="MOVEMENT BOUTS" value={trackPoints.length ? String(bouts) : '—'} unit="bouts" icon={<Waves size={16} />} note={`${trackPoints.length} samples · ${sampleRateHz} Hz`} />
              <Metric label="MEAN BOUT" value={trackPoints.length ? meanBoutSeconds.toFixed(1) : '—'} unit="s" icon={<Clock3 size={16} />} note={`Active time: ${activeSeconds.toFixed(1)} s`} />
            </div>
            {trackPoints.length > 0 && <div className={`signal-hint ${trackingStatus === 'tracked' ? 'signal-good' : ''}`}><Info size={14} /><span>{trackingStatus === 'not_detected' ? 'No moving foreground object was tracked. Check contrast, lighting, and camera stability, then adjust Motion threshold.' : trackingStatus === 'low_coverage' ? 'Track coverage is low. Inspect the trajectory and preview before interpreting the calculated metrics.' : 'One moving foreground object was tracked. Confirm the overlay represents the fish throughout the recording.'} Speeds and distances are in pixels, not calibrated physical units.</span></div>}

            <div className="chart-panel"><div className="chart-heading"><div><h3>Tracked position</h3><span>Detected centroid · percentage of frame</span></div><div className="chart-legend"><span /> X position</div></div>{trackPoints.some((point) => point.tracked) ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartPoints} margin={{ top: 10, right: 14, left: -22, bottom: 0 }}><CartesianGrid stroke="#e9eeeb" vertical={false} /><XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value: number) => formatTime(value)} tickLine={false} axisLine={false} tick={{ fill: '#7f8984', fontSize: 10 }} /><YAxis domain={[0, 100]} reversed tickFormatter={(value: number) => `${value}%`} tickLine={false} axisLine={false} tick={{ fill: '#7f8984', fontSize: 10 }} /><Tooltip labelFormatter={(value) => `${Number(value).toFixed(2)} s`} formatter={(value, name) => [`${value ?? 'miss'}${value == null ? '' : '%'}`, name === 'x_percent' ? 'X position' : 'Y position']} contentStyle={{ border: '1px solid #dce4df', borderRadius: 5, fontSize: 12 }} /><Line type="monotone" dataKey="x_percent" name="x_percent" stroke="#db6d51" strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} /><Line type="monotone" dataKey="y_percent" name="y_percent" stroke="#4a8b77" strokeWidth={1.5} dot={false} connectNulls={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : <div className="chart-empty"><div className="chart-spark"><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /></div><span>No tracked positions. Review track coverage and image contrast.</span></div>}</div>
            <div className="chart-panel"><div className="chart-heading"><div><h3>Movement activity</h3><span>Frame-change signal · {sampleRateHz} samples per second</span></div><div className="chart-legend"><span /> activity</div></div>{trackPoints.length > 1 ? <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartPoints} margin={{ top: 10, right: 14, left: -22, bottom: 0 }}><CartesianGrid stroke="#e9eeeb" vertical={false} /><XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value: number) => formatTime(value)} tickLine={false} axisLine={false} tick={{ fill: '#7f8984', fontSize: 10 }} /><YAxis dataKey="motion" domain={[0, 'auto']} tickFormatter={(value: number) => `${value}%`} tickLine={false} axisLine={false} tick={{ fill: '#7f8984', fontSize: 10 }} /><Tooltip labelFormatter={(value) => `${Number(value).toFixed(2)} s`} formatter={(value) => [`${value}%`, 'Changed pixels']} contentStyle={{ border: '1px solid #dce4df', borderRadius: 5, fontSize: 12 }} /><Line type="monotone" dataKey="motion" stroke="#db6d51" strokeWidth={2} dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : <div className="chart-empty"><div className="chart-spark"><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /></div><span>Movement signal will appear here during analysis.</span></div>}</div>
          </section>

          <aside className="settings-column"><div className="settings-title"><div><span className="settings-icon"><Settings2 size={17} /></span><h2>Analysis setup</h2></div><button className="icon-button" title="Analysis methods" aria-label="Analysis methods" onClick={() => setActiveTab('help')}><CircleHelp size={16} /></button></div>
            <label className="field-label" htmlFor="session-name">SESSION NAME</label><input id="session-name" className="text-input" value={sessionName} onChange={(event) => setSessionName(event.target.value)} />
            <label className="field-label top-gap" htmlFor="tracker-mode">TRACKING METHOD</label><div id="tracker-mode" className="method-card"><span className="method-icon"><Focus size={17} /></span><span><strong>OpenCV foreground track</strong><small>Single moving object · local analysis</small></span><span className="method-status"><Check size={13} /></span></div><p className="method-note">Use a fixed camera and steady lighting. Check track coverage and extracted positions against the recording.</p>
            <div className="settings-divider" />
            <div className="control-heading"><span>Motion threshold</span><span className="control-value">{threshold}</span></div><input className="range-input" type="range" min="1" max="100" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} aria-label="Motion threshold" /><div className="range-captions"><span>More sensitive</span><span>Less sensitive</span></div>
            <div className="control-heading activity-threshold"><span>Active signal threshold</span><span className="control-value">{activityThreshold.toFixed(2)}%</span></div><input className="range-input" type="range" min="0.02" max="8" step="0.02" value={activityThreshold} onChange={(event) => setActivityThreshold(Number(event.target.value))} aria-label="Active signal threshold" /><div className="range-captions"><span>More active samples</span><span>Fewer active samples</span></div>
            <label className="field-label top-gap" htmlFor="frame-rate">VIDEO FRAME RATE</label><div className="input-with-unit"><input id="frame-rate" type="number" min="1" step="1" placeholder="Set source fps" value={frameRate || ''} onChange={(event) => setFrameRate(Math.max(0, Number(event.target.value)))} /><span>fps</span></div><p className="field-footnote">Required for exported frame indices.</p>
            <button className="analyze-button" onClick={startTracking} disabled={!hasSelectedFile || isTracking}><ScanLine size={16} /> {isTracking ? uploadPhase === 'uploading' ? `Uploading video · ${uploadProgress}%` : 'Analyzing locally' : 'Analyze full recording'}</button>
            {notice && <div className="notice-line"><Info size={14} />{notice}</div>}
          </aside>
        </div>}
        {activeTab === 'workspace' && <WorkspaceView workspaces={workspaces} activeWorkspaceId={activeWorkspaceId} sessionCounts={Object.fromEntries(workspaces.map((workspace) => [workspace.id, savedSessions.filter((session) => (session.workspaceId ?? 'research-lab') === workspace.id).length]))} onSelect={selectWorkspace} onCreate={createWorkspace} onDelete={deleteWorkspace} />}
        {activeTab === 'sessions' && <SessionsView sessions={workspaceSessions} onOpen={openSavedSession} onDelete={deleteSavedSession} />}
        {activeTab === 'preferences' && <PreferencesView preferences={{ pixelThreshold: threshold, activityThreshold, frameRate }} onChange={updatePreferences} onReset={resetPreferences} />}
        {activeTab === 'help' && <HelpView onNavigate={setActiveTab} />}
        {activeTab === 'profile' && <ResearchEnvironmentView workspaceName={activeWorkspace.name} sessionCount={workspaceSessions.length} onNavigate={setActiveTab} />}
        <footer className="app-credit">Designed by Andrew G. Watts</footer>
      </main>
    </div>
  )
}

function Metric({ label, value, unit, icon, note }: { label: string; value: string; unit: string; icon: ReactNode; note: string }) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><i>{icon}</i></div><div className="metric-value">{value}<small>{unit}</small></div><div className="metric-note">{note}</div></div>
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return '00:00'
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
  const remainder = Math.floor(seconds % 60).toString().padStart(2, '0')
  return `${minutes}:${remainder}`
}

function formatBytes(bytes: number) {
  return bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GB`
    : bytes >= 1024 ** 2
      ? `${(bytes / 1024 ** 2).toFixed(1)} MB`
      : `${Math.max(1, Math.round(bytes / 1024))} KB`
}

export default App

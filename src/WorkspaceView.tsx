import { Check, FlaskConical, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { ResearchWorkspace } from './workspace-data'

export function WorkspaceView({
  workspaces,
  activeWorkspaceId,
  sessionCounts,
  onSelect,
  onCreate,
  onDelete,
}: {
  workspaces: ResearchWorkspace[]
  activeWorkspaceId: string
  sessionCounts: Record<string, number>
  onSelect: (id: string) => void
  onCreate: (name: string) => void
  onDelete: (id: string) => void
}) {
  const [newName, setNewName] = useState('')

  const submit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const name = newName.trim()
    if (!name) return
    onCreate(name)
    setNewName('')
  }

  return (
    <section className="tab-page workspace-page">
      <div className="tab-page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> LOCAL PROJECTS</div><h2>Workspaces</h2><p>Keep fish video analyses and saved sessions organized by research project.</p></div><span className="session-count">{workspaces.length} {workspaces.length === 1 ? 'workspace' : 'workspaces'}</span></div>
      <div className="workspace-list">
        {workspaces.map((workspace) => {
          const active = workspace.id === activeWorkspaceId
          return <div key={workspace.id} className={`workspace-card ${active ? 'selected' : ''}`}>
            <button className="workspace-card-select" onClick={() => onSelect(workspace.id)} aria-current={active ? 'true' : undefined}>
              <span className="workspace-card-icon"><FlaskConical size={19} /></span>
              <span className="workspace-card-copy"><strong>{workspace.name}</strong><small>{sessionCounts[workspace.id] ?? 0} saved analyses · stored in this browser</small></span>
              {active && <span className="workspace-card-active"><Check size={13} /> Current</span>}
            </button>
            {workspaces.length > 1 && <button className="workspace-delete" title={`Delete ${workspace.name}`} aria-label={`Delete ${workspace.name}`} onClick={() => onDelete(workspace.id)}><Trash2 size={14} /></button>}
          </div>
        })}
      </div>
      <form className="create-workspace" onSubmit={submit}>
        <div><strong>Create a workspace</strong><p>Use a study, cohort, or project name.</p></div>
        <div className="create-workspace-controls"><input value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="e.g. Larval locomotion study" aria-label="New workspace name" maxLength={60} /><button type="submit" disabled={!newName.trim()}><Plus size={15} /> Create</button></div>
      </form>
      <div className="storage-note"><FlaskConical size={15} /><span>Workspaces and session summaries stay in local browser storage. Video files are never included.</span></div>
    </section>
  )
}

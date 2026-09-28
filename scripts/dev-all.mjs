import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { spawn } from 'node:child_process'

const root = process.cwd()
const python = resolve(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')

if (!existsSync(python)) {
  console.error('Python environment not found. Create .venv and install requirements.txt first.')
  process.exit(1)
}

const api = spawn(python, ['-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', '8000'], {
  cwd: root,
  stdio: 'inherit',
})
const web = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], {
  cwd: root,
  stdio: 'inherit',
})

const stop = () => {
  api.kill()
  web.kill()
}

process.on('SIGINT', stop)
process.on('SIGTERM', stop)
api.on('exit', (code) => {
  if (code !== null && code !== 0) process.exitCode = code
  web.kill()
})
web.on('exit', (code) => {
  if (code !== null && code !== 0) process.exitCode = code
  api.kill()
})
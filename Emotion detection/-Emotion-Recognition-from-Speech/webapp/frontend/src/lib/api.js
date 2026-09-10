const BASE = import.meta.env.VITE_API_BASE ?? ''

async function handle(res) {
  const text = await res.text()
  let body
  try {
    body = text ? JSON.parse(text) : {}
  } catch {
    throw new Error(`Server returned invalid JSON (${res.status}).`)
  }
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status}).`)
  return body
}

export const getOverview = () =>
  fetch(`${BASE}/api/overview`).then(handle)

export const getSamples = () =>
  fetch(`${BASE}/api/samples`).then(handle)

export const audioUrl = (id) =>
  `${BASE}/api/audio/${encodeURIComponent(id)}`

export const predictSample = (id) =>
  fetch(`${BASE}/api/predict/sample/${encodeURIComponent(id)}`, {
    method: 'POST',
  }).then(handle)

export function predictFile(file) {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${BASE}/api/predict`, { method: 'POST', body: form }).then(handle)
}

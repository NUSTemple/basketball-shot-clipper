// Thin fetch wrapper - every route registers Flask's shared
// @app.errorhandler(HTTPException) (see app.py), which always responds
// {"error": "<abort message>"} on failure, so ApiError.message is always
// the real, specific reason (e.g. "only the author can edit a label"),
// safe to show directly in the UI instead of a generic "request failed".
export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}) as { error?: string })
    throw new ApiError(res.status, body.error ?? res.statusText)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PATCH', body: JSON.stringify(body ?? {}) }),
  del: (path: string) => request<void>(path, { method: 'DELETE' }),
}

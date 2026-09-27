import { useState } from 'react'

import { ApiError } from '../api/client'
import {
  useAllLabelCategories,
  useCreateLabelCategory,
  useUpdateLabelCategory,
} from '../api/queries/categories'

export function AdminCategoriesPage() {
  const { data: categories } = useAllLabelCategories()
  const createCategory = useCreateLabelCategory()
  const updateCategory = useUpdateLabelCategory()
  const [name, setName] = useState('')

  // The API's 403 is the real permission boundary (see auth.require_admin());
  // this just renders a clean message instead of a raw error toast when a
  // non-admin's mutation is rejected.
  const permissionError =
    (createCategory.error instanceof ApiError && createCategory.error.status === 403) ||
    (updateCategory.error instanceof ApiError && updateCategory.error.status === 403)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    createCategory.mutate(name.trim(), { onSuccess: () => setName('') })
  }

  return (
    <div className="mx-auto max-w-lg p-6">
      <h1 className="mb-4 text-xl font-semibold">Label Categories</h1>

      {permissionError && (
        <p className="mb-3 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800">
          You don't have permission to manage categories.
        </p>
      )}

      <ul className="mb-4 space-y-1">
        {categories?.map((c) => (
          <li key={c.id} className="flex items-center justify-between rounded-md bg-white px-3 py-2 text-sm shadow-sm">
            <span className={c.active ? '' : 'text-slate-400 line-through'}>{c.name}</span>
            <button
              type="button"
              onClick={() => updateCategory.mutate({ id: c.id, active: !c.active })}
              className="text-xs text-orange-600 hover:underline"
            >
              {c.active ? 'Deactivate' : 'Reactivate'}
            </button>
          </li>
        ))}
      </ul>

      <form onSubmit={submit} className="flex gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New category name"
          className="flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
        />
        <button
          type="submit"
          disabled={!name.trim() || createCategory.isPending}
          className="rounded-md bg-orange-500 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          Add
        </button>
      </form>
    </div>
  )
}

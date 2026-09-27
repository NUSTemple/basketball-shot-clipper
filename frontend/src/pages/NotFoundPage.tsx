import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">Not found</h1>
      <Link to="/videos" className="text-orange-600 underline">
        Back to the library
      </Link>
    </div>
  )
}

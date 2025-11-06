"use client"

interface ChartTogglesProps {
  showUncertainty: boolean
  onToggleUncertainty: (value: boolean) => void
}

export default function ChartToggles({ showUncertainty, onToggleUncertainty }: ChartTogglesProps) {
  return (
    <div className="flex items-center gap-3 text-xs text-gray-600 dark:text-gray-300">
      <span className="uppercase tracking-wide">Chart Options</span>
      <button
        onClick={() => onToggleUncertainty(!showUncertainty)}
        className={`px-3 py-1 rounded border ${
          showUncertainty
            ? 'bg-sky-600 text-white border-sky-600'
            : 'bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300'
        }`}
      >
        {showUncertainty ? 'Hide Uncertainty' : 'Show Uncertainty'}
      </button>
    </div>
  )
}


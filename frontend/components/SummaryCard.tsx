"use client"

interface SummaryCardProps {
  result: {
    city: string
    start: string
    horizon: number
    policy: {
      car_free_ratio: number
      renewable_mix: number
      [key: string]: any
    }
    meta: Record<string, any>
  }
}

function formatPercent(value: number | undefined, fallback = "—") {
  if (typeof value !== "number" || Number.isNaN(value)) return fallback
  return `${Math.round(value * 100)}%`
}

function formatBool(value: boolean | undefined) {
  if (typeof value !== "boolean") return "—"
  return value ? "On" : "Off"
}

export default function SummaryCard({ result }: SummaryCardProps) {
  const meta = result.meta ?? {}
  const effective = meta.policy_effective ?? {}
  const generated = meta.generated_at
    ? new Date(meta.generated_at * 1000).toLocaleString()
    : "—"
  const startTime = result.start ? new Date(result.start).toLocaleString() : "—"

  return (
    <div className="bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 rounded-lg p-3 sm:p-4 mb-4 sm:mb-6 text-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-4 mb-3 sm:mb-4">
        <div>
          <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
            Scenario Summary
          </h3>
          <p className="text-gray-600 dark:text-gray-300 text-xs mt-1">
            Simulating {result.city} for {result.horizon} hours starting {startTime}
          </p>
        </div>
        <div className="flex flex-wrap gap-2 sm:gap-3 text-xs text-gray-500 dark:text-gray-400">
          <span className="break-all sm:break-normal">
            Model step: <strong>{meta.model_step ?? "—"}</strong>
          </span>
          <span className="break-all sm:break-normal">
            Hash: <strong>{meta.model_hash ? meta.model_hash.slice(0, 8) : "—"}</strong>
          </span>
          <span className="break-all sm:break-normal">
            Generated: <strong className="whitespace-nowrap">{generated}</strong>
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 text-xs">
        <div className="bg-white/70 dark:bg-gray-900/40 rounded-md p-3 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">Car-Free Ratio</p>
          <p className="text-base font-semibold text-gray-900 dark:text-white">
            {formatPercent(effective.car_free_ratio ?? result.policy.car_free_ratio)}
          </p>
          <p className="text-[11px] text-gray-500 dark:text-gray-400">Effective</p>
        </div>
        <div className="bg-white/70 dark:bg-gray-900/40 rounded-md p-3 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">Renewable Mix</p>
          <p className="text-base font-semibold text-gray-900 dark:text-white">
            {formatPercent(effective.renewable_mix ?? result.policy.renewable_mix)}
          </p>
          <p className="text-[11px] text-gray-500 dark:text-gray-400">Effective</p>
        </div>
        <div className="bg-white/70 dark:bg-gray-900/40 rounded-md p-3 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">Dynamic Pricing</p>
          <p className="text-base font-semibold text-gray-900 dark:text-white">
            {formatBool(effective.dynamic_pricing)}
          </p>
          <p className="text-[11px] text-gray-500 dark:text-gray-400">Policy</p>
        </div>
        <div className="bg-white/70 dark:bg-gray-900/40 rounded-md p-3 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">Economic Activity</p>
          <p className="text-base font-semibold text-gray-900 dark:text-white">
            {(effective.gdp_activity_index ?? 1).toFixed(2)}×
          </p>
          <p className="text-[11px] text-gray-500 dark:text-gray-400">GDP Index</p>
        </div>
      </div>
    </div>
  )
}


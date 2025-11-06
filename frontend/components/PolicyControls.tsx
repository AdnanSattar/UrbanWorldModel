"use client"

import { useEffect, useState } from 'react'

interface PolicyControlsProps {
  onRun: (
    policy: {
      car_free_ratio: number
      renewable_mix: number
      dynamic_pricing?: boolean
      stricter_car_bans?: boolean
      gdp_activity_index?: number
      commercial_load_factor?: number
    },
    horizonHours: number
  ) => void
  loading?: boolean
  city?: string
}

export default function PolicyControls({ onRun, loading = false, city = 'Lahore' }: PolicyControlsProps) {
  const [carFree, setCarFree] = useState(0.1)
  const [renewMix, setRenewMix] = useState(0.2)
  const [horizon, setHorizon] = useState(48)
  const [dynamicPricing, setDynamicPricing] = useState(false)
  const [stricterBans, setStricterBans] = useState(false)
  const [gdpIndex, setGdpIndex] = useState(1.0) // 1.0 = baseline
  const [commercialLoad, setCommercialLoad] = useState(1.0)

  // Load/save presets per city
  useEffect(() => {
    try {
      const raw = localStorage.getItem(`urbansim/presets/${city}`)
      if (raw) {
        const p = JSON.parse(raw)
        if (typeof p.car_free_ratio === 'number') setCarFree(p.car_free_ratio)
        if (typeof p.renewable_mix === 'number') setRenewMix(p.renewable_mix)
        if (typeof p.horizon_hours === 'number') setHorizon(p.horizon_hours)
      }
    } catch {}
  }, [city])

  useEffect(() => {
    try {
      localStorage.setItem(
        `urbansim/presets/${city}`,
        JSON.stringify({ car_free_ratio: carFree, renewable_mix: renewMix, horizon_hours: horizon })
      )
    } catch {}
  }, [city, carFree, renewMix, horizon])

  const handleRun = () => {
    onRun(
      {
        car_free_ratio: carFree,
        renewable_mix: renewMix,
        dynamic_pricing: dynamicPricing,
        stricter_car_bans: stricterBans,
        gdp_activity_index: gdpIndex,
        commercial_load_factor: commercialLoad
      },
      horizon
    )
  }

  return (
    <div className="space-y-5 sm:space-y-6">
      {/* Car-Free Ratio Slider */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Car-Free Ratio
          </label>
          <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
            {Math.round(carFree * 100)}%
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={carFree}
          onChange={(e) => setCarFree(parseFloat(e.target.value))}
          className="w-full h-2 sm:h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-blue-600 touch-manipulation"
          disabled={loading}
        />
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Percentage of vehicles removed from roads
        </p>
      </div>

      {/* Renewable Mix Slider */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Renewable Energy Mix
          </label>
          <span className="text-sm font-semibold text-green-600 dark:text-green-400">
            {Math.round(renewMix * 100)}%
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={renewMix}
          onChange={(e) => setRenewMix(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-green-600"
          disabled={loading}
        />
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Fraction of energy from renewable sources
        </p>
      </div>

      {/* Horizon Slider */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Horizon (hours)
          </label>
          <span className="text-sm font-semibold text-indigo-600 dark:text-indigo-400">
            {horizon}
          </span>
        </div>
        <input
          type="range"
          min="1"
          max="168"
          step="1"
          value={horizon}
          onChange={(e) => setHorizon(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-indigo-600"
          disabled={loading}
        />
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Forecast horizon (1–168 hours)
        </p>
      </div>

      {/* Policy Coupling Toggles */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        <label className="flex items-center gap-2.5 sm:gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer touch-manipulation min-h-[44px] sm:min-h-0">
          <input
            type="checkbox"
            checked={dynamicPricing}
            onChange={(e) => setDynamicPricing(e.target.checked)}
            disabled={loading}
            className="w-5 h-5 sm:w-4 sm:h-4 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
          />
          <span>Dynamic Demand Pricing</span>
        </label>
        <label className="flex items-center gap-2.5 sm:gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer touch-manipulation min-h-[44px] sm:min-h-0">
          <input
            type="checkbox"
            checked={stricterBans}
            onChange={(e) => setStricterBans(e.target.checked)}
            disabled={loading}
            className="w-5 h-5 sm:w-4 sm:h-4 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
          />
          <span>Stricter Car Bans</span>
        </label>
      </div>

      {/* Economic Exogenous Inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              GDP Activity Index
            </label>
            <span className="text-sm font-semibold text-indigo-600 dark:text-indigo-400">
              {gdpIndex.toFixed(2)}x
            </span>
          </div>
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.01"
            value={gdpIndex}
            onChange={(e) => setGdpIndex(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-indigo-600"
            disabled={loading}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Proxy of economic activity (0.5×–1.5× baseline)
          </p>
        </div>
        <div>
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Commercial Load Factor
            </label>
            <span className="text-sm font-semibold text-indigo-600 dark:text-indigo-400">
              {commercialLoad.toFixed(2)}x
            </span>
          </div>
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.01"
            value={commercialLoad}
            onChange={(e) => setCommercialLoad(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-indigo-600"
            disabled={loading}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Commercial demand intensity (0.5×–1.5× baseline)
          </p>
        </div>
      </div>

      {/* Preset Scenarios */}
      <div>
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Quick Presets:
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => { setCarFree(0); setRenewMix(0); }}
            className="px-3 py-1 text-xs bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded transition-colors"
            disabled={loading}
          >
            Baseline
          </button>
          <button
            onClick={() => { setCarFree(0.2); setRenewMix(0); }}
            className="px-3 py-1 text-xs bg-blue-100 dark:bg-blue-900 hover:bg-blue-200 dark:hover:bg-blue-800 rounded transition-colors"
            disabled={loading}
          >
            Car-Free Days
          </button>
          <button
            onClick={() => { setCarFree(0); setRenewMix(0.5); }}
            className="px-3 py-1 text-xs bg-green-100 dark:bg-green-900 hover:bg-green-200 dark:hover:bg-green-800 rounded transition-colors"
            disabled={loading}
          >
            Green Energy
          </button>
          <button
            onClick={() => { setCarFree(0.3); setRenewMix(0.6); }}
            className="px-3 py-1 text-xs bg-purple-100 dark:bg-purple-900 hover:bg-purple-200 dark:hover:bg-purple-800 rounded transition-colors"
            disabled={loading}
          >
            Aggressive
          </button>
        </div>
      </div>

      {/* Run Button */}
      <button
        onClick={handleRun}
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:bg-gray-400 text-white font-semibold py-3 sm:py-2.5 px-6 rounded-lg transition-colors shadow-md hover:shadow-lg disabled:cursor-not-allowed touch-manipulation min-h-[48px] sm:min-h-0 text-base sm:text-sm"
      >
        {loading ? 'Running Simulation...' : 'Run Simulation'}
      </button>
    </div>
  )
}


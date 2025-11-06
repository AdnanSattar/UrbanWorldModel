"use client"

import { useState, useEffect } from 'react'
import PolicyControls from '@/components/PolicyControls'
import AirQualityChart from '@/components/AirQualityChart'
import EnergyChart from '@/components/EnergyChart'
import TrafficChart from '@/components/TrafficChart'
import SummaryCard from '@/components/SummaryCard'
import KpiDeck from '@/components/KpiDeck'
import DistrictCompare from '@/components/DistrictCompare'
import ChartToggles from '@/components/ChartToggles'
import LatentViz from '@/components/LatentViz'
import ScenarioSaver from '@/components/ScenarioSaver'
import GoalSeek from '@/components/GoalSeek'
import dynamic from 'next/dynamic'
import axios from 'axios'
import type { SavedScenario } from '@/lib/scenarioStorage'
const MapView = dynamic(() => import('@/components/MapView'), { ssr: false })

interface PolicyConfig {
  car_free_ratio: number
  renewable_mix: number
  dynamic_pricing?: boolean
  stricter_car_bans?: boolean
  gdp_activity_index?: number
  commercial_load_factor?: number
}

interface SimulationDataPoint {
  hour: number
  pm25: number
  energy_mwh: number
  traffic_index: number
  pm25_std?: number
  energy_std?: number
  traffic_std?: number
}

interface SimulationResult {
  city: string
  start: string
  horizon: number
  policy: PolicyConfig
  simulated: SimulationDataPoint[]
  meta: {
    generated_at: number
    model_version?: string
    note?: string
    model_step?: number
    model_hash?: string
    checkpoint_file?: string
    policy_effective?: {
      car_free_ratio?: number
      renewable_mix?: number
      dynamic_pricing?: boolean
      stricter_car_bans?: boolean
      gdp_activity_index?: number
      commercial_load_factor?: number
    }
  }
}

export default function SimulatorPage() {
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [city, setCity] = useState('Lahore')
  const [cursor, setCursor] = useState<number>(48)
  const [playing, setPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(1)
  const [events, setEvents] = useState<Array<{ label: string; hour: number }>>([])
  const [showUncertainty, setShowUncertainty] = useState(true)

  const runSimulation = async (policy: PolicyConfig, horizonHours: number, cityOverride?: string) => {
    setLoading(true)
    setError(null)
    const targetCity = cityOverride ?? city
    if (cityOverride && cityOverride !== city) {
      setCity(cityOverride)
    }

    const payload = {
      city: targetCity,
      start_time: new Date().toISOString(),
      horizon_hours: horizonHours,
      policy,
      uncertainty_samples: 20
    }

    try {
      // Use relative path - Next.js rewrites will proxy to backend
      const res = await axios.post(`/api/simulate`, payload)
      setResult(res.data)
    } catch (err: any) {
      console.error('Simulation error:', err)
      setError(err.response?.data?.detail || err.message || 'Failed to run simulation')
    } finally {
      setLoading(false)
    }
  }

  const suggestPolicy = async () => {
    try {
      const res = await axios.post(`/api/optimize`, {
        city,
        horizon_hours: 48,
        weights: { pm25: 1.0, energy: 0.3, traffic: 0.3 }
      })
      const best = res.data?.policy || { car_free_ratio: 0.2, renewable_mix: 0.4 }
      await runSimulation(best as any, 48)
    } catch (e) {
      console.error('Optimize failed', e)
    }
  }

  const handleLoadScenario = async (scenario: SavedScenario) => {
    await runSimulation(scenario.policy, scenario.horizon, scenario.city)
  }

  const handleGoalSeekApply = (policy: PolicyConfig) => {
    runSimulation(policy, result?.horizon ?? 48, city)
  }

  // reset playback when result changes
  useEffect(() => {
    if (!result) {
      setEvents([])
      return
    }
    setCursor(Math.max(0, result.horizon - 1))
    setPlaying(false)
    setPlaySpeed(1)

    const data = result.simulated
    if (!data.length) {
      setEvents([])
      return
    }
    const pmMaxIndex = data.reduce((maxIdx, point, idx, arr) => (point.pm25 > arr[maxIdx].pm25 ? idx : maxIdx), 0)
    const pmMinIndex = data.reduce((minIdx, point, idx, arr) => (point.pm25 < arr[minIdx].pm25 ? idx : minIdx), 0)
    const energyMinIndex = data.reduce((minIdx, point, idx, arr) => (point.energy_mwh < arr[minIdx].energy_mwh ? idx : minIdx), 0)
    const trafficMinIndex = data.reduce((minIdx, point, idx, arr) => (point.traffic_index < arr[minIdx].traffic_index ? idx : minIdx), 0)
    const candidates = [
      { label: 'PM2.5 Peak', hour: pmMaxIndex },
      { label: 'PM2.5 Trough', hour: pmMinIndex },
      { label: 'Energy Low', hour: energyMinIndex },
      { label: 'Traffic Relief', hour: trafficMinIndex },
    ]
    const unique = new Map<number, { label: string; hour: number }>()
    candidates.forEach((ev) => {
      if (!unique.has(ev.hour)) unique.set(ev.hour, ev)
    })
    setEvents(Array.from(unique.values()))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  // playback timer with speed control
  useEffect(() => {
    if (!result) return
    if (!playing) return
    const delay = Math.max(100, 400 / Math.max(0.25, playSpeed))
    const id = setInterval(() => {
      setCursor((c) => {
        const maxH = result.horizon - 1
        return c >= maxH ? 0 : c + 1
      })
    }, delay)
    return () => clearInterval(id)
  }, [playing, result, playSpeed])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6">
          <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-gray-900 dark:text-white leading-tight">
            UrbanSim WM — Smart City World Model
          </h1>
          <p className="mt-1 sm:mt-2 text-xs sm:text-sm text-gray-600 dark:text-gray-300">
            Simulate urban dynamics: energy, air quality, and mobility
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6 md:py-8">
        {/* Policy Controls Section */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6 mb-6 sm:mb-8">
          <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            Policy Controls
          </h2>
          <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-300 mb-4 sm:mb-6">
            Adjust policy parameters to simulate different urban interventions
          </p>
          {/* City Selector */}
          <div className="mb-4 sm:mb-6 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
              City
            </label>
            <select
              value={city}
              onChange={(e) => setCity(e.target.value)}
              className="w-full sm:w-auto rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="Lahore">Lahore</option>
              <option value="Karachi">Karachi</option>
              <option value="Islamabad">Islamabad</option>
              <option value="Peshawar">Peshawar</option>
              <option value="Quetta">Quetta</option>
            </select>
          </div>
          <PolicyControls onRun={runSimulation} loading={loading} city={city} />
          <div className="mt-4">
            <button
              onClick={suggestPolicy}
              className="w-full bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white font-semibold py-2.5 sm:py-2 px-4 rounded-lg transition-colors touch-manipulation"
            >
              Suggest Policy
            </button>
          </div>
          {/* Replay Controls */}
          {result && (
            <div className="mt-4 sm:mt-6 space-y-3">
            <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <button
                onClick={() => setPlaying((p) => !p)}
                className="px-4 py-2 sm:px-3 sm:py-1 rounded-lg bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white text-sm font-medium transition-colors touch-manipulation min-h-[44px] sm:min-h-0"
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <span className="text-sm text-gray-600 dark:text-gray-300 whitespace-nowrap">t = {cursor} h</span>
              <div className="flex items-center gap-1.5 sm:gap-2 text-xs text-gray-500 dark:text-gray-400">
                <span className="hidden sm:inline">Speed</span>
                <span className="sm:hidden">Spd</span>
                {[0.5, 1, 2].map((speed) => (
                  <button
                    key={speed}
                    onClick={() => setPlaySpeed(speed)}
                    className={`px-2.5 sm:px-2 py-1.5 sm:py-1 rounded border text-xs min-w-[36px] sm:min-w-0 transition-colors touch-manipulation ${
                      playSpeed === speed
                        ? 'bg-indigo-600 text-white border-indigo-600'
                        : 'bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:border-indigo-400'
                    }`}
                  >
                    {speed}x
                  </button>
                ))}
              </div>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, (result?.horizon ?? 1) - 1)}
              value={cursor}
              onChange={(e) => setCursor(parseInt(e.target.value))}
              className="w-full h-2 accent-indigo-600 cursor-pointer"
            />
            {events.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                <span className="hidden sm:inline">Jump to:</span>
                <span className="sm:hidden">Jump:</span>
                {events.map((ev) => (
                  <button
                    key={`${ev.label}-${ev.hour}`}
                    onClick={() => {
                      setCursor(ev.hour)
                      setPlaying(false)
                    }}
                    className="px-2.5 py-1.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded transition-colors touch-manipulation text-xs"
                  >
                    {ev.label} (h{ev.hour})
                  </button>
                ))}
              </div>
            )}
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-8">
            <h3 className="text-red-800 dark:text-red-200 font-semibold">Error</h3>
            <p className="text-red-600 dark:text-red-300 text-sm">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-8 mb-8 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-blue-800 dark:text-blue-200 font-medium">
              Running simulation...
            </p>
          </div>
        )}

        {/* Results Section */}
        {result && !loading && (
          <>
            <SummaryCard result={result} />
            <KpiDeck result={result} />
            <div className="mb-6">
              <ChartToggles
                showUncertainty={showUncertainty}
                onToggleUncertainty={setShowUncertainty}
              />
            </div>
            {/* Metadata */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6 mb-6 sm:mb-8">
              <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                Simulation Results
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">City:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {result.city}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Horizon:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {result.horizon} hours
                  </span>
                </div>
                <div className="sm:col-span-2 md:col-span-1">
                  <span className="text-gray-600 dark:text-gray-400">Policy:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white break-words">
                    {Math.round(result.policy.car_free_ratio * 100)}% car-free, {Math.round(result.policy.renewable_mix * 100)}% renewable
                  </span>
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="space-y-6 sm:space-y-8">
              {/* Air Quality Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                  Air Quality (PM2.5)
                </h3>
                <div className="w-full overflow-x-auto -mx-2 sm:mx-0 px-2 sm:px-0">
                  <AirQualityChart
                    data={(result.simulated).slice(0, Math.min(cursor + 1, result.horizon))}
                    showUncertainty={showUncertainty}
                  />
                </div>
              </div>

              {/* Energy Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                  Energy Consumption
                </h3>
                <div className="w-full overflow-x-auto -mx-2 sm:mx-0 px-2 sm:px-0">
                  <EnergyChart
                    data={(result.simulated).slice(0, Math.min(cursor + 1, result.horizon))}
                    showUncertainty={showUncertainty}
                  />
                </div>
              </div>

              {/* Traffic Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                  Traffic Congestion
                </h3>
                <div className="w-full overflow-x-auto -mx-2 sm:mx-0 px-2 sm:px-0">
                  <TrafficChart data={(result.simulated).slice(0, Math.min(cursor + 1, result.horizon))} />
                </div>
              </div>
            </div>

            {/* Geospatial View */}
            <div className="mt-6 sm:mt-8 grid gap-6 sm:gap-8 lg:grid-cols-2">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                  City Heatmap
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                  Synthetic PM2.5 overlay for demonstration. Real ETL integration will replace this grid.
                </p>
                <div className="w-full overflow-x-auto">
                  <MapView city={result.city} data={result.simulated} cursor={cursor} />
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                  District Comparison
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                  Relative PM2.5, energy, and traffic intensity by district at the current timestep.
                </p>
                <div className="w-full overflow-x-auto">
                  <DistrictCompare city={result.city} data={result.simulated} cursor={cursor} />
                </div>
              </div>
            </div>

            {/* Model Explainability */}
            <div className="mt-6 sm:mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6">
              <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-gray-900 dark:text-white">
                Latent Dynamics Projection
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                Demo view of latent RSSM states projected into 2D via synthetic UMAP. Integrate training exports for real data.
              </p>
              <div className="w-full overflow-x-auto">
                <LatentViz city={result.city} />
              </div>
            </div>

            <GoalSeek city={city} onApply={handleGoalSeekApply} />

            <ScenarioSaver result={result} onLoadScenario={handleLoadScenario} />

            {/* Raw Data (collapsible) */}
            <details className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6 mt-6 sm:mt-8">
              <summary className="cursor-pointer text-base sm:text-lg font-semibold text-gray-900 dark:text-white touch-manipulation">
                View Raw Data (JSON)
              </summary>
              <pre className="mt-4 bg-gray-100 dark:bg-gray-900 p-3 sm:p-4 rounded overflow-x-auto text-xs sm:text-sm">
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </>
        )}

        {/* Instructions (show when no results) */}
        {!result && !loading && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 sm:p-8 text-center">
            <h3 className="text-base sm:text-lg font-semibold mb-2 text-gray-900 dark:text-white">
              Get Started
            </h3>
            <p className="text-sm sm:text-base text-gray-600 dark:text-gray-300">
              Adjust the policy sliders above and click "Run Simulation" to see results
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-8 sm:mt-12 md:mt-16 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6">
          <p className="text-center text-xs sm:text-sm text-gray-600 dark:text-gray-400">
            UrbanSim WM v0.1.0 — World Model for Smart Cities
          </p>
        </div>
      </footer>
    </div>
  )
}


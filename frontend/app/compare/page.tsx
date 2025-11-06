"use client"

import { useEffect, useState } from 'react'
import Link from 'next/link'
import axios from 'axios'
import SummaryCard from '@/components/SummaryCard'
import KpiDeck from '@/components/KpiDeck'
import AirQualityChart from '@/components/AirQualityChart'
import EnergyChart from '@/components/EnergyChart'
import TrafficChart from '@/components/TrafficChart'
import { listScenarios, SavedScenario } from '@/lib/scenarioStorage'

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
  meta: Record<string, any>
}

const initialState: SimulationResult | null = null

export default function ComparePage() {
  const [scenarios, setScenarios] = useState<SavedScenario[]>([])
  const [leftId, setLeftId] = useState<string>('')
  const [rightId, setRightId] = useState<string>('')
  const [leftResult, setLeftResult] = useState<SimulationResult | null>(initialState)
  const [rightResult, setRightResult] = useState<SimulationResult | null>(initialState)
  const [loadingSide, setLoadingSide] = useState<'left' | 'right' | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setScenarios(listScenarios())
  }, [])

  const runScenario = async (scenario: SavedScenario, side: 'left' | 'right') => {
    setLoadingSide(side)
    setError(null)
    try {
      const res = await axios.post('/api/simulate', {
        city: scenario.city,
        start_time: new Date().toISOString(),
        horizon_hours: scenario.horizon,
        policy: scenario.policy,
        uncertainty_samples: 10,
      })
      if (side === 'left') {
        setLeftResult(res.data)
      } else {
        setRightResult(res.data)
      }
    } catch (err: any) {
      console.error(err)
      setError(err.response?.data?.detail || err.message || 'Failed to run scenario')
    } finally {
      setLoadingSide(null)
    }
  }

  const onSelectScenario = (id: string, side: 'left' | 'right') => {
    if (!id) return
    const scenario = scenarios.find((s) => s.id === id)
    if (!scenario) return
    if (side === 'left') setLeftId(id)
    else setRightId(id)
    runScenario(scenario, side)
  }

  const renderPanel = (result: SimulationResult | null, label: string) => {
    if (!result) {
      return <p className="text-sm text-gray-500 dark:text-gray-400">Select a scenario to load results.</p>
    }
    return (
      <div className="space-y-4">
        <SummaryCard result={result} />
        <KpiDeck result={result} />
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-900 dark:text-white">Air Quality</h4>
          <AirQualityChart data={result.simulated} showUncertainty={false} />
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-900 dark:text-white">Energy</h4>
          <EnergyChart data={result.simulated} showUncertainty={false} />
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-900 dark:text-white">Traffic</h4>
          <TrafficChart data={result.simulated} />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="bg-white dark:bg-gray-900 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Scenario Comparison</h1>
            <p className="text-sm text-gray-600 dark:text-gray-300">Compare saved interventions side-by-side.</p>
          </div>
          <Link href="/" className="text-sm text-indigo-600 dark:text-indigo-400">← Back to simulator</Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-700 dark:text-red-200">{error}</p>
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <ScenarioSelect
            label="Scenario A"
            scenarios={scenarios}
            value={leftId}
            loading={loadingSide === 'left'}
            onChange={(id) => onSelectScenario(id, 'left')}
          />
          <ScenarioSelect
            label="Scenario B"
            scenarios={scenarios}
            value={rightId}
            loading={loadingSide === 'right'}
            onChange={(id) => onSelectScenario(id, 'right')}
          />
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div>{renderPanel(leftResult, 'Scenario A')}</div>
          <div>{renderPanel(rightResult, 'Scenario B')}</div>
        </div>
      </main>
    </div>
  )
}

function ScenarioSelect({
  label,
  scenarios,
  value,
  loading,
  onChange,
}: {
  label: string
  scenarios: SavedScenario[]
  value: string
  loading: boolean
  onChange: (id: string) => void
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">{label}</p>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-900 dark:text-white px-3 py-2"
      >
        <option value="">Select saved scenario…</option>
        {scenarios.map((scenario) => (
          <option key={scenario.id} value={scenario.id}>
            {scenario.label} · {scenario.city}
          </option>
        ))}
      </select>
      {loading && <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">Loading…</p>}
    </div>
  )
}


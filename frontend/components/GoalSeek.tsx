"use client"

import { useState } from 'react'
import axios from 'axios'

interface PolicyConfig {
  car_free_ratio: number
  renewable_mix: number
  dynamic_pricing?: boolean
  stricter_car_bans?: boolean
  gdp_activity_index?: number
  commercial_load_factor?: number
}

interface GoalSeekProps {
  city: string
  onApply: (policy: PolicyConfig) => void
}

export default function GoalSeek({ city, onApply }: GoalSeekProps) {
  const [targetPm25, setTargetPm25] = useState(35)
  const [budget, setBudget] = useState(32)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{
    policy: PolicyConfig
    score: number
    tested: number
  } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const runGoalSeek = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post('/api/optimize', {
        city,
        horizon_hours: 48,
        weights: { pm25: 1.0, energy: 0.3, traffic: 0.3 },
        pm25_target: targetPm25,
        eval_budget: budget,
      })
      setResult({
        policy: response.data.policy,
        score: response.data.score,
        tested: response.data.candidates_tested,
      })
    } catch (err: any) {
      console.error(err)
      setError(err.response?.data?.detail || err.message || 'Goal seek failed')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mt-8">
      <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">Goal-Seeking Assistant</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
        Target a PM2.5 level and let the optimizer suggest an intervention mix.
      </p>
      <div className="grid sm:grid-cols-2 gap-3 mb-4 text-sm">
        <label className="flex flex-col">
          <span className="text-gray-600 dark:text-gray-300 mb-1">PM2.5 Target (µg/m³)</span>
          <input
            type="number"
            min={5}
            max={150}
            value={targetPm25}
            onChange={(e) => setTargetPm25(parseFloat(e.target.value) || 0)}
            className="rounded border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
          />
        </label>
        <label className="flex flex-col">
          <span className="text-gray-600 dark:text-gray-300 mb-1">Evaluation Budget</span>
          <input
            type="number"
            min={4}
            max={200}
            value={budget}
            onChange={(e) => setBudget(parseInt(e.target.value) || 4)}
            className="rounded border border-gray-300 dark:border-gray-600 px-3 py-2 bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
          />
        </label>
      </div>
      <div className="flex flex-wrap gap-3">
        <button
          onClick={runGoalSeek}
          disabled={loading}
          className="px-4 py-2 rounded bg-sky-600 hover:bg-sky-700 text-white text-sm disabled:bg-gray-400"
        >
          {loading ? 'Searching…' : 'Find Policy'}
        </button>
        {result && (
          <button
            onClick={() => onApply(result.policy)}
            className="px-4 py-2 rounded border border-gray-300 dark:border-gray-600 text-sm text-gray-700 dark:text-gray-300"
          >
            Apply to Controls
          </button>
        )}
      </div>
      {error && (
        <p className="text-xs text-red-600 dark:text-red-300 mt-3">{error}</p>
      )}
      {result && !error && (
        <div className="mt-4 text-xs text-gray-600 dark:text-gray-300 space-y-1">
          <p>
            Suggested mix: <strong>{Math.round(result.policy.car_free_ratio * 100)}% car-free</strong>, {' '}
            <strong>{Math.round(result.policy.renewable_mix * 100)}% renewables</strong>
          </p>
          <p>
            Composite score: {result.score.toFixed(1)} · Evaluations used: {result.tested}
          </p>
        </div>
      )}
    </div>
  )
}


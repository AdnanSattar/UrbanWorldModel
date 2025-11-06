"use client"

import { useEffect, useState } from 'react'
import { deleteScenario, listScenarios, saveScenario, SavedScenario } from '@/lib/scenarioStorage'

interface ScenarioSaverProps {
  result: {
    city: string
    horizon: number
    policy: {
      car_free_ratio: number
      renewable_mix: number
      dynamic_pricing?: boolean
      stricter_car_bans?: boolean
      gdp_activity_index?: number
      commercial_load_factor?: number
    }
  }
  onLoadScenario: (scenario: SavedScenario) => void
}

export default function ScenarioSaver({ result, onLoadScenario }: ScenarioSaverProps) {
  const [label, setLabel] = useState('')
  const [scenarios, setScenarios] = useState<SavedScenario[]>([])

  useEffect(() => {
    setScenarios(listScenarios())
  }, [])

  const handleSave = () => {
    if (!result) return
    const trimmed = label.trim() || `${result.city} @ ${new Date().toLocaleTimeString()}`
    const saved = saveScenario({
      label: trimmed,
      city: result.city,
      horizon: result.horizon,
      policy: result.policy,
    })
    setScenarios(listScenarios())
    setLabel('')
    onLoadScenario(saved)
  }

  const handleDelete = (id: string) => {
    deleteScenario(id)
    setScenarios(listScenarios())
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mt-8">
      <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">Scenario Library</h3>
      <div className="flex flex-col md:flex-row gap-3 md:items-center mb-4">
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Label this scenario"
          className="flex-1 rounded border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
        />
        <button
          onClick={handleSave}
          className="px-4 py-2 rounded bg-emerald-600 text-white text-sm"
        >
          Save &amp; Run Later
        </button>
      </div>

      {scenarios.length === 0 ? (
        <p className="text-xs text-gray-500 dark:text-gray-400">No saved scenarios yet.</p>
      ) : (
        <ul className="space-y-2 text-sm">
          {scenarios.map((scenario) => (
            <li
              key={scenario.id}
              className="flex flex-col md:flex-row md:items-center justify-between gap-2 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-2"
            >
              <div>
                <p className="font-medium text-gray-900 dark:text-white">{scenario.label}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {scenario.city} · {scenario.horizon}h · Car-free {Math.round(scenario.policy.car_free_ratio * 100)}% / Renewables {Math.round(scenario.policy.renewable_mix * 100)}%
                </p>
              </div>
              <div className="flex gap-2 text-xs">
                <button
                  onClick={() => onLoadScenario(scenario)}
                  className="px-3 py-1 rounded bg-indigo-600 text-white"
                >
                  Load
                </button>
                <button
                  onClick={() => handleDelete(scenario.id)}
                  className="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300"
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}


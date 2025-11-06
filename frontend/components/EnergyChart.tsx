"use client"

import { AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'

interface DataPoint {
  hour: number
  pm25: number
  energy_mwh: number
  traffic_index: number
  energy_std?: number
}

interface EnergyChartProps {
  data: DataPoint[]
  showUncertainty?: boolean
}

export default function EnergyChart({ data, showUncertainty = true }: EnergyChartProps) {
  const totalEnergy = data.reduce((sum, d) => sum + d.energy_mwh, 0)
  const avgEnergy = totalEnergy / data.length
  const peakEnergy = Math.max(...data.map(d => d.energy_mwh))
  const minEnergy = Math.min(...data.map(d => d.energy_mwh))

  return (
    <div>
      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6 text-sm">
        <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Total</p>
          <p className="text-xl font-bold text-purple-700 dark:text-purple-400">
            {totalEnergy.toFixed(0)}
          </p>
          <p className="text-xs text-gray-500">MWh</p>
        </div>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Average</p>
          <p className="text-xl font-bold text-blue-700 dark:text-blue-400">
            {avgEnergy.toFixed(0)}
          </p>
          <p className="text-xs text-gray-500">MWh/hour</p>
        </div>
        <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Minimum</p>
          <p className="text-xl font-bold text-green-700 dark:text-green-400">
            {minEnergy.toFixed(0)}
          </p>
          <p className="text-xs text-gray-500">MWh</p>
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Peak</p>
          <p className="text-xl font-bold text-red-700 dark:text-red-400">
            {peakEnergy.toFixed(0)}
          </p>
          <p className="text-xs text-gray-500">MWh</p>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="colorEnergy" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="hour"
            label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
            stroke="#666"
          />
          <YAxis
            label={{ value: 'Energy (MWh)', angle: -90, position: 'insideLeft' }}
            stroke="#666"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
            formatter={(value: number) => [`${value.toFixed(2)} MWh`, 'Energy']}
            labelFormatter={(hour) => `Hour ${hour}`}
          />
          <Legend />
          {showUncertainty && (
            <>
              <Area
                type="monotone"
                dataKey={(d: any) => d.energy_mwh + (d.energy_std ? 2 * d.energy_std : 0)}
                stroke="none"
                fill="#8b5cf6"
                fillOpacity={0.1}
                name="Energy +2σ"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey={(d: any) => d.energy_mwh - (d.energy_std ? 2 * d.energy_std : 0)}
                stroke="none"
                fill="#8b5cf6"
                fillOpacity={0.1}
                name="Energy -2σ"
                isAnimationActive={false}
              />
            </>
          )}
          <Area
            type="monotone"
            dataKey="energy_mwh"
            stroke="#8b5cf6"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorEnergy)"
            name="Energy Consumption"
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Additional Info */}
      <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded text-xs">
        <p className="text-gray-600 dark:text-gray-400">
          <strong>Note:</strong> Energy consumption varies throughout the day based on residential,
          commercial, and industrial demand patterns. Policy interventions like renewable energy
          adoption can reduce overall carbon emissions.
        </p>
      </div>
    </div>
  )
}


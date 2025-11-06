"use client"

import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, Area } from 'recharts'

interface DataPoint {
  hour: number
  pm25: number
  energy_mwh: number
  traffic_index: number
  pm25_std?: number
}

interface AirQualityChartProps {
  data: DataPoint[]
  showUncertainty?: boolean
}

export default function AirQualityChart({ data, showUncertainty = true }: AirQualityChartProps) {
  // Air quality thresholds (WHO guidelines)
  const getAQICategory = (pm25: number): string => {
    if (pm25 <= 12) return 'Good'
    if (pm25 <= 35.4) return 'Moderate'
    if (pm25 <= 55.4) return 'Unhealthy for Sensitive'
    if (pm25 <= 150.4) return 'Unhealthy'
    if (pm25 <= 250.4) return 'Very Unhealthy'
    return 'Hazardous'
  }

  const avgPM25 = data.reduce((sum, d) => sum + d.pm25, 0) / data.length
  const minPM25 = Math.min(...data.map(d => d.pm25))
  const maxPM25 = Math.max(...data.map(d => d.pm25))

  return (
    <div>
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6 text-sm">
        <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Minimum</p>
          <p className="text-xl font-bold text-green-700 dark:text-green-400">
            {minPM25.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500">µg/m³</p>
        </div>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Average</p>
          <p className="text-xl font-bold text-blue-700 dark:text-blue-400">
            {avgPM25.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500">{getAQICategory(avgPM25)}</p>
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Maximum</p>
          <p className="text-xl font-bold text-red-700 dark:text-red-400">
            {maxPM25.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500">µg/m³</p>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="hour"
            label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
            stroke="#666"
          />
          <YAxis
            label={{ value: 'PM2.5 (µg/m³)', angle: -90, position: 'insideLeft' }}
            stroke="#666"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
            formatter={(value: number) => [`${value.toFixed(2)} µg/m³`, 'PM2.5']}
            labelFormatter={(hour) => `Hour ${hour}`}
          />
          <Legend />
          {showUncertainty && (
            <>
              <Area
                type="monotone"
                dataKey={(d: any) => d.pm25 + (d.pm25_std ? 2 * d.pm25_std : 0)}
                stroke="none"
                fill="#ff7300"
                fillOpacity={0.1}
                name="PM2.5 +2σ"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey={(d: any) => d.pm25 - (d.pm25_std ? 2 * d.pm25_std : 0)}
                stroke="none"
                fill="#ff7300"
                fillOpacity={0.1}
                name="PM2.5 -2σ"
                isAnimationActive={false}
              />
            </>
          )}
          <Line
            type="monotone"
            dataKey="pm25"
            stroke="#ff7300"
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            name="PM2.5 Level"
          />
          {/* Reference line for WHO guideline (15 µg/m³ 24-hour mean) */}
          <Line
            type="monotone"
            dataKey={() => 15}
            stroke="#22c55e"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="WHO Guideline"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}


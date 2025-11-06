"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, Cell, AreaChart, Area } from 'recharts'

interface DataPoint {
  hour: number
  pm25: number
  energy_mwh: number
  traffic_index: number
  traffic_std?: number
}

interface TrafficChartProps {
  data: DataPoint[]
}

export default function TrafficChart({ data }: TrafficChartProps) {
  const avgTraffic = data.reduce((sum, d) => sum + d.traffic_index, 0) / data.length
  const peakTraffic = Math.max(...data.map(d => d.traffic_index))
  const minTraffic = Math.min(...data.map(d => d.traffic_index))

  // Color bars based on congestion level
  const getBarColor = (value: number): string => {
    if (value < 0.5) return '#22c55e' // Green - free flow
    if (value < 0.8) return '#eab308' // Yellow - moderate
    if (value < 1.2) return '#f97316' // Orange - congested
    return '#ef4444' // Red - heavily congested
  }

  const getCongestionLevel = (value: number): string => {
    if (value < 0.5) return 'Free Flow'
    if (value < 0.8) return 'Moderate'
    if (value < 1.2) return 'Congested'
    return 'Heavy Congestion'
  }

  return (
    <div>
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6 text-sm">
        <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Minimum</p>
          <p className="text-xl font-bold text-green-700 dark:text-green-400">
            {minTraffic.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500">{getCongestionLevel(minTraffic)}</p>
        </div>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Average</p>
          <p className="text-xl font-bold text-blue-700 dark:text-blue-400">
            {avgTraffic.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500">{getCongestionLevel(avgTraffic)}</p>
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded">
          <p className="text-gray-600 dark:text-gray-400">Peak</p>
          <p className="text-xl font-bold text-red-700 dark:text-red-400">
            {peakTraffic.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500">{getCongestionLevel(peakTraffic)}</p>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="hour"
            label={{ value: 'Hours', position: 'insideBottom', offset: -5 }}
            stroke="#666"
          />
          <YAxis
            label={{ value: 'Traffic Index', angle: -90, position: 'insideLeft' }}
            domain={[0, 2]}
            stroke="#666"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
            formatter={(value: number) => [
              `${value.toFixed(3)} (${getCongestionLevel(value)})`,
              'Traffic Index'
            ]}
            labelFormatter={(hour) => `Hour ${hour}`}
          />
          <Legend />
          <Bar dataKey="traffic_index" name="Traffic Congestion" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.traffic_index)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend for colors */}
      <div className="mt-4 flex justify-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-green-500 rounded"></div>
          <span className="text-gray-600 dark:text-gray-400">Free Flow (&lt;0.5)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-yellow-500 rounded"></div>
          <span className="text-gray-600 dark:text-gray-400">Moderate (0.5-0.8)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-orange-500 rounded"></div>
          <span className="text-gray-600 dark:text-gray-400">Congested (0.8-1.2)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-500 rounded"></div>
          <span className="text-gray-600 dark:text-gray-400">Heavy (&gt;1.2)</span>
        </div>
      </div>

      {/* Additional Info */}
      <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded text-xs">
        <p className="text-gray-600 dark:text-gray-400">
          <strong>Traffic Index:</strong> 0 = no traffic, 1 = normal flow, &gt;1 = congestion.
          Car-free policies and improved public transit can significantly reduce congestion.
        </p>
      </div>
    </div>
  )
}


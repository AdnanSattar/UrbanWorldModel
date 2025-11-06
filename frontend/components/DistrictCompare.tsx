"use client"

import { useMemo } from 'react'
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from 'recharts'
import { getCityGrid } from '@/lib/geo/mockGrid'

interface DistrictCompareProps {
  city: string
  data: Array<{ pm25: number; energy_mwh: number; traffic_index: number }>
  cursor: number
}

export default function DistrictCompare({ city, data, cursor }: DistrictCompareProps) {
  const grid = useMemo(() => getCityGrid(city), [city])
  const current = data[Math.max(0, Math.min(data.length - 1, cursor))]

  if (!current) {
    return <p className="text-sm text-gray-500 dark:text-gray-400">No data available.</p>
  }

  const dataset = grid.map((cell) => ({
    name: cell.name,
    pm25: Number((current.pm25 * cell.weight).toFixed(1)),
    energy: Number((current.energy_mwh * (0.8 + cell.weight * 0.3)).toFixed(1)),
    traffic: Number((current.traffic_index * (0.9 + (cell.weight - 1) * 0.4)).toFixed(2)),
  }))

  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={dataset}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={60} />
          <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} />
          <Tooltip formatter={(value: number, name: string) => {
            if (name === 'pm25') return [`${value} µg/m³`, 'PM2.5']
            if (name === 'energy') return [`${value} MWh`, 'Energy']
            return [`${value}`, 'Traffic Index']
          }} />
          <Legend wrapperStyle={{ fontSize: '11px' }} />
          <Bar dataKey="pm25" fill="#ef4444" name="PM2.5" radius={[4, 4, 0, 0]} />
          <Bar dataKey="energy" fill="#3b82f6" name="Energy (MWh)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="traffic" fill="#10b981" name="Traffic Index" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}


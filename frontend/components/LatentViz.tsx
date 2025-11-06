"use client"

import { useEffect, useState } from 'react'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, ZAxis } from 'recharts'

interface LatentPoint {
  x: number
  y: number
  label: string
  weight: number
}

interface LatentVizProps {
  city: string
}

export default function LatentViz({ city }: LatentVizProps) {
  const [points, setPoints] = useState<LatentPoint[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    const fetchData = async () => {
      setLoading(true)
      try {
        const res = await fetch(`/api/explain/latent_sample?city=${encodeURIComponent(city)}`, {
          signal: controller.signal,
        })
        if (!res.ok) throw new Error('Failed to load latent sample')
        const json = await res.json()
        setPoints(json.points ?? [])
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          console.error(err)
          setPoints([])
        }
      } finally {
        setLoading(false)
      }
    }
    fetchData()
    return () => controller.abort()
  }, [city])

  if (loading) {
    return <p className="text-sm text-gray-500 dark:text-gray-400">Loading latent projection…</p>
  }

  if (!points.length) {
    return <p className="text-sm text-gray-500 dark:text-gray-400">Latent representation unavailable.</p>
  }

  const grouped = points.reduce<Record<string, LatentPoint[]>>((acc, point) => {
    acc[point.label] = acc[point.label] || []
    acc[point.label].push(point)
    return acc
  }, {})

  const colors = ['#1d4ed8', '#ef4444', '#22c55e', '#f97316']
  const series = Object.entries(grouped)

  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <XAxis type="number" dataKey="x" name="Latent X" domain={[-1.5, 1.5]} tick={{ fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="Latent Y" domain={[-1.5, 1.5]} tick={{ fontSize: 11 }} />
          <ZAxis type="number" dataKey="weight" range={[80, 180]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          {series.map(([label, pts], idx) => (
            <Scatter
              key={label}
              name={label}
              data={pts}
              fill={colors[idx % colors.length]}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}


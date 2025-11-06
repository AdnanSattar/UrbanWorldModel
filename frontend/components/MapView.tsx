"use client"

import { useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet'
import { getCityCenter, getCityGrid } from '@/lib/geo/mockGrid'

interface MapViewProps {
  city: string
  data: Array<{ pm25: number }>
  cursor: number
}

const markerRadius = 18

export default function MapView({ city, data, cursor }: MapViewProps) {
  if (typeof window === 'undefined') return null

  const grid = useMemo(() => getCityGrid(city), [city])
  const center = useMemo(() => getCityCenter(city), [city])
  const sample = data[Math.max(0, Math.min(data.length - 1, cursor))]
  const pm25 = sample?.pm25 ?? 0

  return (
    <MapContainer
      center={center}
      zoom={11}
      scrollWheelZoom={false}
      style={{ height: 320, width: '100%' }}
      className="rounded-lg overflow-hidden"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {grid.map((cell) => {
        const intensity = pm25 * cell.weight
        const color = intensityToColor(intensity)
        return (
          <CircleMarker
            key={cell.id}
            center={[cell.lat, cell.lng]}
            pathOptions={{ color, fillColor: color, fillOpacity: 0.55 }}
            radius={markerRadius}
          >
            <Tooltip direction="top" offset={[0, -markerRadius]} opacity={1}
              className="bg-white text-xs p-1 rounded shadow">
              <div className="text-gray-800">
                <p className="font-semibold">{cell.name}</p>
                <p>{intensity.toFixed(1)} µg/m³</p>
              </div>
            </Tooltip>
          </CircleMarker>
        )
      })}
    </MapContainer>
  )
}

function intensityToColor(value: number) {
  const clamped = Math.min(200, Math.max(0, value))
  const ratio = clamped / 200
  const r = Math.floor(255 * ratio)
  const g = Math.floor(100 * (1 - ratio))
  const b = Math.floor(80 * (1 - ratio))
  return `rgb(${r}, ${g}, ${b})`
}


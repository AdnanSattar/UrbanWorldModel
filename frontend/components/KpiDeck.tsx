"use client"

import { useEffect, useState } from "react"

interface KpiDeckProps {
  result: {
    horizon: number
    policy: { car_free_ratio: number; renewable_mix: number }
    simulated: Array<{
      pm25: number
      energy_mwh: number
      traffic_index: number
    }>
    meta: Record<string, any>
  }
}

const baselineEnergyPerHour = 1200
const co2Factor = 0.4 // tonnes per MWh avoided (rough heuristic)
const energyCostPerMWh = 110 // USD equivalent heuristic

function AnimatedNumber({ value, suffix = "", decimals = 1 }: { value: number; suffix?: string; decimals?: number }) {
  const [display, setDisplay] = useState(value)

  useEffect(() => {
    let frame: number
    const startValue = display
    const delta = value - startValue
    const duration = 500
    let startTime: number | null = null

    const animate = (timestamp: number) => {
      if (startTime === null) startTime = timestamp
      const progress = Math.min((timestamp - startTime) / duration, 1)
      setDisplay(startValue + delta * progress)
      if (progress < 1) frame = requestAnimationFrame(animate)
    }

    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  return (
    <span>
      {display.toFixed(decimals)}{suffix}
    </span>
  )
}

export default function KpiDeck({ result }: KpiDeckProps) {
  const points = result.simulated
  if (!points.length) return null

  const avgPm25 = points.reduce((sum, p) => sum + p.pm25, 0) / points.length
  const totalEnergy = points.reduce((sum, p) => sum + p.energy_mwh, 0)
  const avgTraffic = points.reduce((sum, p) => sum + p.traffic_index, 0) / points.length

  const effectiveRenewable = result.meta?.policy_effective?.renewable_mix ?? result.policy.renewable_mix
  const effectiveCarFree = result.meta?.policy_effective?.car_free_ratio ?? result.policy.car_free_ratio

  const baselineEnergy = baselineEnergyPerHour * result.horizon
  const energyDifference = baselineEnergy - totalEnergy
  const energyDeltaPercent = (energyDifference / baselineEnergy) * 100
  const co2SavedTonnes = Math.max(0, effectiveRenewable * baselineEnergy * co2Factor)
  const costReduction = energyDifference * energyCostPerMWh * 1e-6 // convert to million USD
  const trafficSpeedIndex = Math.max(0, Math.min(100, (2 - avgTraffic) / 2 * 100))
  const carFreeImpact = effectiveCarFree * 100

  const kpis = [
    {
      label: "Average PM2.5",
      value: avgPm25,
      suffix: " µg/m³",
      decimals: 1,
      description: "Mean concentration over horizon",
    },
    {
      label: "Estimated CO₂ Saved",
      value: co2SavedTonnes,
      suffix: " t",
      decimals: 0,
      description: `Approximate tonnes avoided via ${Math.round(effectiveRenewable * 100)}% renewables`,
    },
    {
      label: "Energy Cost Δ",
      value: energyDeltaPercent,
      suffix: "%",
      decimals: 1,
      description: `${energyDifference > 0 ? "Reduction" : "Increase"} vs. baseline (ₘ ${costReduction.toFixed(2)} USD)`,
    },
    {
      label: "Traffic Speed Index",
      value: trafficSpeedIndex,
      suffix: "%",
      decimals: 1,
      description: `Higher is better. Car-free impact ${carFreeImpact.toFixed(0)}%`,
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {kpis.map((kpi) => (
        <div
          key={kpi.label}
          className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 shadow-sm"
        >
          <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">
            {kpi.label}
          </p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-white">
            <AnimatedNumber value={kpi.value} suffix={kpi.suffix} decimals={kpi.decimals} />
          </p>
          <p className="mt-2 text-[11px] text-gray-500 dark:text-gray-400 leading-snug">
            {kpi.description}
          </p>
        </div>
      ))}
    </div>
  )
}


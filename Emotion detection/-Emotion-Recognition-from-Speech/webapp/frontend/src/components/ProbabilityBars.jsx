import { motion, useReducedMotion } from 'framer-motion'
import { emotionColor, emotionLabel } from '../lib/emotions'

/**
 * Full probability distribution, sorted high to low.
 *
 * Every row carries the emotion name and its exact percentage, so the colour
 * is decoration rather than the only way to read the value.
 */
export default function ProbabilityBars({ probabilities, isDark, predicted }) {
  const reduced = useReducedMotion()
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1])

  return (
    <ul className="space-y-2.5" aria-label="Emotion probabilities">
      {rows.map(([emotion, p], i) => {
        const color = emotionColor(emotion, isDark)
        const isTop = emotion === predicted
        return (
          <li key={emotion}>
            <div className="mb-1 flex items-baseline justify-between gap-3">
              <span
                className={`text-sm ${
                  isTop ? 'font-semibold text-fg' : 'font-medium text-muted'
                }`}
              >
                {emotionLabel(emotion)}
              </span>
              <span
                className={`tnum text-sm ${
                  isTop ? 'font-semibold text-fg' : 'text-muted'
                }`}
              >
                {(p * 100).toFixed(2)}%
              </span>
            </div>
            <div
              className="h-2 w-full overflow-hidden rounded-full bg-surface-3"
              role="meter"
              aria-valuenow={Number((p * 100).toFixed(2))}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${emotionLabel(emotion)} probability`}
            >
              <motion.div
                className="h-full rounded-full"
                style={{ background: color, opacity: isTop ? 1 : 0.55 }}
                initial={reduced ? false : { width: 0 }}
                animate={{ width: `${Math.max(p * 100, 0.6)}%` }}
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: 'spring', stiffness: 120, damping: 20, delay: 0.05 + i * 0.045 }
                }
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

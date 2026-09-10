import { motion, useReducedMotion } from 'framer-motion'

const SIZE = 168
const STROKE = 12
const R = (SIZE - STROKE) / 2
const CIRC = 2 * Math.PI * R

/**
 * Radial confidence gauge.
 *
 * The threshold tick is drawn on the arc so the reading is not just "how
 * confident" but "confident enough to trust, or not" — the threshold was
 * calibrated on validation data, and the ring shows where it sits.
 */
export default function ConfidenceRing({ value, threshold, color, reliable }) {
  const reduced = useReducedMotion()
  const offset = CIRC * (1 - value)
  const tickAngle = threshold * 360 - 90

  return (
    <div className="relative" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} role="img"
           aria-label={`Confidence ${(value * 100).toFixed(1)} percent, threshold ${(threshold * 100).toFixed(0)} percent`}>
        <g transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}>
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={R}
            fill="none" stroke="var(--surface-3)" strokeWidth={STROKE}
          />
          <motion.circle
            cx={SIZE / 2} cy={SIZE / 2} r={R}
            fill="none" stroke={color} strokeWidth={STROKE}
            strokeLinecap="round" strokeDasharray={CIRC}
            initial={reduced ? false : { strokeDashoffset: CIRC }}
            animate={{ strokeDashoffset: offset }}
            transition={reduced ? { duration: 0 } : { duration: 1, ease: [0.22, 0.61, 0.36, 1] }}
          />
        </g>
        {/* Calibrated threshold marker */}
        <line
          x1={SIZE / 2 + (R - STROKE / 2 - 3) * Math.cos((tickAngle * Math.PI) / 180)}
          y1={SIZE / 2 + (R - STROKE / 2 - 3) * Math.sin((tickAngle * Math.PI) / 180)}
          x2={SIZE / 2 + (R + STROKE / 2 + 3) * Math.cos((tickAngle * Math.PI) / 180)}
          y2={SIZE / 2 + (R + STROKE / 2 + 3) * Math.sin((tickAngle * Math.PI) / 180)}
          stroke="var(--fg)" strokeWidth={2} strokeLinecap="round" opacity={0.55}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="tnum text-3xl font-semibold tracking-tight text-fg"
          initial={reduced ? false : { opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          {(value * 100).toFixed(1)}%
        </motion.span>
        <span className="mt-0.5 text-[11px] font-medium uppercase tracking-wider text-faint">
          confidence
        </span>
        <span
          className="mt-1 text-[11px] font-medium"
          style={{ color: reliable ? 'var(--ok)' : 'var(--warn)' }}
        >
          {reliable ? 'above threshold' : 'below threshold'}
        </span>
      </div>
    </div>
  )
}

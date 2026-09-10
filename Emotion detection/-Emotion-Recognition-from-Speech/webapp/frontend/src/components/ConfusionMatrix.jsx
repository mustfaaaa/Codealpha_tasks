import { motion, useReducedMotion } from 'framer-motion'
import { useState } from 'react'
import { EMOTION_ORDER, emotionLabel } from '../lib/emotions'
import { Card } from './ui'

/**
 * Confusion matrix as a heatmap.
 *
 * Chart-accessibility rule for heatmaps: never rely on colour alone. Every
 * cell prints its count, the diagonal is outlined, and a keyboard-focusable
 * cell announces its full "true X predicted Y" reading.
 */
export default function ConfusionMatrix({ matrix, classes = EMOTION_ORDER }) {
  const reduced = useReducedMotion()
  const [hover, setHover] = useState(null)
  const rowTotals = matrix.map((r) => r.reduce((a, b) => a + b, 0))
  const max = Math.max(...matrix.flat())

  return (
    <Card className="p-5">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-fg">Confusion matrix — test split</h3>
        <p className="text-xs text-muted">
          rows = true emotion · columns = prediction · cells = clip count
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-separate border-spacing-[3px]">
          <caption className="sr-only">
            Confusion matrix of the selected model on 240 held-out test clips.
          </caption>
          <thead>
            <tr>
              <th className="w-20" />
              {classes.map((c) => (
                <th
                  key={c}
                  scope="col"
                  className="pb-2 text-[10px] font-medium uppercase tracking-wide text-faint"
                >
                  <span className="block -rotate-45 origin-center whitespace-nowrap">
                    {c.slice(0, 5)}
                  </span>
                </th>
              ))}
              <th className="pb-2 pl-2 text-[10px] font-medium uppercase text-faint">
                recall
              </th>
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={classes[i]}>
                <th
                  scope="row"
                  className="pr-2 text-right text-xs font-medium text-muted"
                >
                  {emotionLabel(classes[i])}
                </th>
                {row.map((v, j) => {
                  const isDiag = i === j
                  const intensity = max ? v / max : 0
                  const active = hover?.[0] === i && hover?.[1] === j
                  return (
                    <td key={j} className="p-0">
                      <motion.div
                        tabIndex={0}
                        role="img"
                        aria-label={`True ${classes[i]}, predicted ${classes[j]}: ${v} clips`}
                        onMouseEnter={() => setHover([i, j])}
                        onMouseLeave={() => setHover(null)}
                        onFocus={() => setHover([i, j])}
                        onBlur={() => setHover(null)}
                        initial={reduced ? false : { opacity: 0, scale: 0.9 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{
                          duration: 0.3,
                          delay: reduced ? 0 : (i + j) * 0.012,
                        }}
                        className="flex aspect-square min-h-9 cursor-default items-center justify-center rounded text-xs font-medium"
                        style={{
                          background: v === 0
                            ? 'var(--surface-3)'
                            : `color-mix(in srgb, ${isDiag ? 'var(--ok)' : 'var(--primary)'} ${12 + intensity * 78}%, var(--surface-3))`,
                          color: intensity > 0.45 ? '#fff' : 'var(--muted)',
                          outline: active ? '2px solid var(--accent)' : 'none',
                          outlineOffset: 1,
                        }}
                      >
                        {v || ''}
                      </motion.div>
                    </td>
                  )
                })}
                <td className="pl-2 text-right">
                  <span className="tnum text-xs font-semibold text-fg">
                    {rowTotals[i] ? ((row[i] / rowTotals[i]) * 100).toFixed(0) : 0}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-line pt-3 text-xs text-muted">
        <span className="flex items-center gap-1.5">
          <span className="size-3 rounded" style={{ background: 'var(--ok)' }} aria-hidden="true" />
          correct (diagonal)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="size-3 rounded" style={{ background: 'var(--primary)' }} aria-hidden="true" />
          confusion — darker means more clips
        </span>
      </div>
    </Card>
  )
}

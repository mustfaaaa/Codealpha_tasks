import { motion, useReducedMotion } from 'framer-motion'
import { useState } from 'react'
import { ArrowRight, Cpu, Gauge, Target, Zap } from 'lucide-react'
import { emotionColor, emotionLabel } from '../lib/emotions'
import ConfusionMatrix from './ConfusionMatrix'
import { Card, Chip, CountUp, EASE, Hint, Reveal, SectionHeading, Stat } from './ui'

export default function Dashboard({ data, isDark }) {
  const { test, model, comparison, confusionMatrix, topConfusions, perActor, calibration } = data

  return (
    <section className="border-t border-line bg-surface-2/40">
      <div className="mx-auto max-w-6xl px-5 py-20 sm:py-24">
        <SectionHeading
          id="performance"
          eyebrow="Measured results"
          title="How well does it actually work?"
          description="Every number here comes from one evaluation on the untouched test split — 240 clips from four actors the model never heard. Nothing was tuned against it."
        />

        {/* headline metrics */}
        <div className="mt-10 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat
            label="Accuracy"
            value={<CountUp value={test.accuracy * 100} decimals={1} suffix="%" />}
            sub="8-way · chance is 12.5%"
            accent
          />
          <Stat
            label={<Hint text="Averages per-class F1 without weighting by support, so the small neutral class counts as much as any other.">Macro F1</Hint>}
            value={<CountUp value={test.macroF1} decimals={3} />}
            sub="primary selection metric"
          />
          <Stat
            label="ROC-AUC"
            value={<CountUp value={test.rocAuc} decimals={3} />}
            sub="one-vs-rest, macro"
          />
          <Stat
            label="Inference"
            value={<CountUp value={test.msPerClip} decimals={1} suffix=" ms" />}
            sub="per 3-second clip, CPU"
          />
        </div>

        {/* model comparison */}
        <div className="mt-14 grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          <Reveal>
            <ModelComparison rows={comparison} selected={model.name} />
          </Reveal>
          <Reveal delay={0.08}>
            <SelectionCard model={model} calibration={calibration} />
          </Reveal>
        </div>

        {/* per class */}
        <div className="mt-14 grid gap-6 lg:grid-cols-2">
          <Reveal>
            <PerClass perClass={test.perClass} isDark={isDark} />
          </Reveal>
          <Reveal delay={0.08}>
            <ConfusionMatrix matrix={confusionMatrix} classes={model.classes} />
          </Reveal>
        </div>

        {/* error analysis */}
        <div className="mt-14 grid gap-6 lg:grid-cols-2">
          <Reveal>
            <Confusions items={topConfusions} isDark={isDark} />
          </Reveal>
          <Reveal delay={0.08}>
            <PerActor rows={perActor} />
          </Reveal>
        </div>
      </div>
    </section>
  )
}

/* -------------------------------------------------------------- comparison */
function ModelComparison({ rows, selected }) {
  const reduced = useReducedMotion()
  const [metric, setMetric] = useState('Macro F1')
  const metrics = ['Macro F1', 'Accuracy', 'Weighted F1']
  const sorted = [...rows].sort((a, b) => b[metric] - a[metric])
  const max = Math.max(...sorted.map((r) => r[metric]))

  return (
    <Card className="p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-fg">Every model we trained</h3>
        <div className="flex gap-1 rounded-lg border border-line bg-surface p-1">
          {metrics.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMetric(m)}
              className={`relative cursor-pointer rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                metric === m ? 'text-primary-fg' : 'text-muted hover:text-fg'
              }`}
            >
              {metric === m && (
                <motion.span
                  layoutId="metric-pill"
                  className="absolute inset-0 rounded-md bg-primary"
                  transition={{ type: 'spring', stiffness: 320, damping: 30 }}
                />
              )}
              <span className="relative z-10">{m}</span>
            </button>
          ))}
        </div>
      </div>

      <ul className="space-y-2.5">
        {sorted.map((r, i) => {
          const isSel = r.Model === selected
          const v = r[metric]
          return (
            <li key={r.Model}>
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="flex items-center gap-2 truncate text-[13px]">
                  <span className={isSel ? 'font-semibold text-fg' : 'text-muted'}>
                    {r.Model}
                  </span>
                  {isSel && <Chip tone="accent">selected</Chip>}
                </span>
                <span className={`tnum text-[13px] ${isSel ? 'font-semibold text-fg' : 'text-muted'}`}>
                  {v.toFixed(3)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: isSel ? 'var(--primary)' : 'var(--line-strong)' }}
                  initial={reduced ? false : { width: 0 }}
                  whileInView={{ width: `${(v / max) * 100}%` }}
                  viewport={{ once: true }}
                  transition={reduced ? { duration: 0 } : { duration: 0.7, ease: EASE, delay: i * 0.05 }}
                />
              </div>
            </li>
          )
        })}
      </ul>

      <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-muted">
        The BiLSTM lands <em>below</em> the logistic-regression baseline on this
        metric. With ~960 utterances, local spectral texture generalises better
        than raw frame-by-frame temporal modelling — added complexity did not pay.
      </p>
    </Card>
  )
}

/* --------------------------------------------------------------- selection */
function SelectionCard({ model, calibration }) {
  const s = model.selection
  return (
    <Card className="flex h-full flex-col p-5">
      <h3 className="text-sm font-semibold text-fg">Why this model was chosen</h3>

      <div className="mt-4 space-y-3">
        <Row icon={Target} label="Best raw score"
             value={`${s.best_raw_model} · ${s.best_raw_val_macro_f1.toFixed(4)}`} />
        <Row icon={Gauge} label="Bootstrap standard error"
             value={`± ${s.bootstrap_se.toFixed(4)} on 240 val clips`} />
        <Row icon={Cpu} label="Statistically tied"
             value={s.models_within_1se.join(', ')} />
        <Row icon={Zap} label="Selected (fewest params)"
             value={`${model.name} · ${model.params.toLocaleString()} params`} highlight />
      </div>

      <p className="mt-4 text-xs leading-relaxed text-muted">
        Raw argmax would have picked <span className="font-medium text-fg">{s.best_raw_model}</span>,
        but its lead is about a third of one standard error — roughly two clips.
        Under the one-standard-error rule the simplest tied model wins, which is
        4× smaller and 3.7× faster. The test set later confirmed the tie was real.
      </p>

      <div className="mt-auto rounded-lg border border-line bg-surface p-3.5">
        <p className="text-xs font-medium text-fg">Calibrated confidence gate</p>
        <p className="mt-1 text-xs leading-relaxed text-muted">
          Threshold <span className="tnum font-medium text-fg">{calibration.threshold.toFixed(2)}</span> was
          swept on validation data, not invented. It accepts{' '}
          <span className="tnum font-medium text-fg">{(calibration.coverage_at_threshold * 100).toFixed(0)}%</span>{' '}
          of clips at{' '}
          <span className="tnum font-medium text-fg">{(calibration.selective_accuracy_at_threshold * 100).toFixed(0)}%</span>{' '}
          selective accuracy.
        </p>
      </div>
    </Card>
  )
}

function Row({ icon: Icon, label, value, highlight }) {
  return (
    <div className="flex items-start gap-2.5">
      <span className={`mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md ${
        highlight ? 'bg-primary text-primary-fg' : 'bg-surface-3 text-muted'
      }`}>
        <Icon className="size-3.5" aria-hidden="true" />
      </span>
      <div className="min-w-0">
        <p className="text-[11px] uppercase tracking-wide text-faint">{label}</p>
        <p className={`text-[13px] ${highlight ? 'font-semibold text-fg' : 'text-muted'}`}>
          {value}
        </p>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- per class */
function PerClass({ perClass, isDark }) {
  const reduced = useReducedMotion()
  const rows = Object.entries(perClass).sort((a, b) => b[1]['f1-score'] - a[1]['f1-score'])
  return (
    <Card className="p-5">
      <h3 className="mb-1 text-sm font-semibold text-fg">Per-emotion F1</h3>
      <p className="mb-4 text-xs text-muted">
        Disgust and calm are reliable; happy and angry are where the model struggles.
      </p>
      <ul className="space-y-3">
        {rows.map(([emo, m], i) => {
          const color = emotionColor(emo, isDark)
          return (
            <li key={emo}>
              <div className="mb-1 flex items-baseline justify-between text-[13px]">
                <span className="flex items-center gap-2 font-medium text-fg">
                  <span className="size-2.5 rounded-full" style={{ background: color }} aria-hidden="true" />
                  {emotionLabel(emo)}
                </span>
                <span className="tnum text-muted">
                  P {m.precision.toFixed(2)} · R {m.recall.toFixed(2)} ·{' '}
                  <span className="font-semibold text-fg">F1 {m['f1-score'].toFixed(2)}</span>
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: color }}
                  initial={reduced ? false : { width: 0 }}
                  whileInView={{ width: `${m['f1-score'] * 100}%` }}
                  viewport={{ once: true }}
                  transition={reduced ? { duration: 0 } : { duration: 0.7, ease: EASE, delay: i * 0.05 }}
                />
              </div>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}

/* --------------------------------------------------------------- confusions */
function Confusions({ items, isDark }) {
  return (
    <Card className="p-5">
      <h3 className="mb-1 text-sm font-semibold text-fg">What gets confused with what</h3>
      <p className="mb-4 text-xs leading-relaxed text-muted">
        The errors are not random — they cluster by <strong className="text-fg">arousal</strong>.
        Energy and pitch range make high-arousal emotions easy to tell from low-arousal
        ones, but say little about valence. Distinguishing excitement from anger is
        exactly what the model misses.
      </p>
      <ul className="space-y-2">
        {items.map((c, i) => (
          <motion.li
            key={`${c.true}-${c.predicted}`}
            initial={{ opacity: 0, x: -10 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.35, delay: i * 0.05, ease: EASE }}
            className="flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2"
          >
            <span className="flex items-center gap-1.5 text-[13px] font-medium text-fg">
              <span className="size-2 rounded-full"
                    style={{ background: emotionColor(c.true, isDark) }} aria-hidden="true" />
              {emotionLabel(c.true)}
            </span>
            <ArrowRight className="size-3.5 text-faint" aria-hidden="true" />
            <span className="flex items-center gap-1.5 text-[13px] text-muted">
              <span className="size-2 rounded-full"
                    style={{ background: emotionColor(c.predicted, isDark) }} aria-hidden="true" />
              {emotionLabel(c.predicted)}
            </span>
            <span className="tnum ml-auto text-xs font-semibold text-fg">
              {c.count} clips
            </span>
          </motion.li>
        ))}
      </ul>
    </Card>
  )
}

/* ---------------------------------------------------------------- per actor */
function PerActor({ rows }) {
  const reduced = useReducedMotion()
  return (
    <Card className="p-5">
      <h3 className="mb-1 text-sm font-semibold text-fg">Accuracy per test speaker</h3>
      <p className="mb-4 text-xs leading-relaxed text-muted">
        Four unseen actors, 60 clips each. The spread is real: a single headline
        number hides meaningful speaker-to-speaker variance.
      </p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {rows.map((r, i) => (
          <motion.div
            key={r.actor}
            initial={reduced ? false : { opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.06, ease: EASE }}
            className="rounded-lg border border-line bg-surface p-3 text-center"
          >
            <p className="text-[11px] uppercase tracking-wide text-faint">
              actor {r.actor}
            </p>
            <p className="tnum mt-1 text-xl font-semibold text-fg">
              {(r.mean * 100).toFixed(1)}%
            </p>
            <p className="text-[11px] text-muted">{r.count} clips</p>
          </motion.div>
        ))}
      </div>
    </Card>
  )
}

import { motion, useReducedMotion } from 'framer-motion'
import { ArrowDown, AudioLines, ShieldCheck } from 'lucide-react'
import { EASE } from './ui'

/** Static bar heights — deterministic, so the waveform never reflows on render. */
const BARS = [
  18, 34, 52, 30, 66, 88, 54, 72, 96, 60, 38, 74, 100, 46, 28, 58,
  82, 40, 64, 92, 50, 26, 44, 78, 56, 34, 68, 86, 42, 22, 36, 62,
]

export default function Hero({ dataset, test }) {
  const reduced = useReducedMotion()

  return (
    <header className="relative overflow-hidden border-b border-line">
      <div className="grid-bg absolute inset-0" aria-hidden="true" />
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-72 opacity-60"
        style={{
          background:
            'radial-gradient(60% 100% at 50% 0%, color-mix(in srgb, var(--primary) 16%, transparent), transparent 70%)',
        }}
        aria-hidden="true"
      />

      <div className="relative mx-auto max-w-6xl px-5 pb-16 pt-20 sm:pb-20 sm:pt-28">
        <motion.div
          initial={reduced ? false : 'hidden'}
          animate="show"
          variants={{ hidden: {}, show: { transition: { staggerChildren: 0.08 } } }}
          className="max-w-3xl"
        >
          <motion.div
            variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE } } }}
            className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface-2 px-3 py-1.5"
          >
            <ShieldCheck className="size-3.5 text-primary" aria-hidden="true" />
            <span className="text-xs font-medium text-muted">
              Speaker-independent evaluation · no actor shared across splits
            </span>
          </motion.div>

          <motion.h1
            variants={{ hidden: { opacity: 0, y: 18 }, show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: EASE } } }}
            className="text-4xl font-semibold leading-[1.08] tracking-tight text-fg sm:text-6xl"
          >
            Reading emotion
            <br />
            from the sound of speech.
          </motion.h1>

          <motion.p
            variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE } } }}
            className="mt-5 max-w-xl text-base leading-relaxed text-muted sm:text-lg"
          >
            A convolutional network trained on MFCC features from RAVDESS,
            classifying eight emotions from voices it has never heard before.
            Upload a clip and watch it work — then inspect exactly how well it does.
          </motion.p>

          <motion.div
            variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE } } }}
            className="mt-8 flex flex-wrap gap-3"
          >
            <a
              href="#analyze"
              className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-fg transition-transform duration-200 hover:-translate-y-0.5"
            >
              <AudioLines className="size-4" aria-hidden="true" />
              Analyse a clip
            </a>
            <a
              href="#performance"
              className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-surface-2 px-5 py-2.5 text-sm font-medium text-fg transition-colors duration-200 hover:border-line-strong"
            >
              See the numbers
              <ArrowDown className="size-4" aria-hidden="true" />
            </a>
          </motion.div>
        </motion.div>

        {/* animated waveform */}
        <div
          className="mt-14 flex h-24 items-center justify-center gap-[3px] sm:h-32"
          aria-hidden="true"
        >
          {BARS.map((h, i) => (
            <motion.span
              key={i}
              className="w-1.5 rounded-full sm:w-2"
              style={{
                background: i % 4 === 0 ? 'var(--primary)' : 'var(--line-strong)',
                opacity: i % 4 === 0 ? 0.85 : 0.5,
              }}
              initial={reduced ? { height: `${h}%` } : { height: '4%' }}
              animate={
                reduced
                  ? { height: `${h}%` }
                  : { height: [`${h * 0.35}%`, `${h}%`, `${h * 0.5}%`] }
              }
              transition={
                reduced
                  ? { duration: 0 }
                  : {
                      duration: 2.4 + (i % 5) * 0.35,
                      repeat: Infinity,
                      repeatType: 'mirror',
                      ease: 'easeInOut',
                      delay: i * 0.035,
                    }
              }
            />
          ))}
        </div>

        {/* key facts */}
        <div className="mt-10 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
          <Fact value={`${(test.accuracy * 100).toFixed(1)}%`} label="test accuracy" note="unseen speakers" />
          <Fact value={test.macroF1.toFixed(3)} label="macro F1" note="all 8 classes equally" />
          <Fact value={dataset.speakers} label="actors" note={`${dataset.recordings} recordings`} />
          <Fact value={dataset.emotions} label="emotions" note="RAVDESS speech set" />
        </div>
      </div>
    </header>
  )
}

function Fact({ value, label, note }) {
  return (
    <div className="bg-surface px-4 py-4">
      <p className="tnum text-2xl font-semibold tracking-tight text-fg sm:text-3xl">
        {value}
      </p>
      <p className="mt-0.5 text-xs font-medium text-fg">{label}</p>
      <p className="text-[11px] text-faint">{note}</p>
    </div>
  )
}

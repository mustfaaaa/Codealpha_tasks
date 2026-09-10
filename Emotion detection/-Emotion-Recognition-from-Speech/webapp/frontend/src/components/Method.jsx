import { motion, useReducedMotion } from 'framer-motion'
import { AudioWaveform, Boxes, GitBranch, Layers, Scissors, SlidersHorizontal } from 'lucide-react'
import { Card, EASE, Reveal, SectionHeading } from './ui'

const STEPS = [
  {
    icon: Scissors,
    title: 'Preprocess',
    body: 'Mono, resampled to 16 kHz, leading and trailing silence trimmed, peak-normalised, then centre-cropped or padded to exactly 3 seconds.',
    detail: 'identical for training and inference',
  },
  {
    icon: AudioWaveform,
    title: 'Extract MFCCs',
    body: '40 MFCCs plus their delta and delta-delta, giving a 40 × 94 × 3 tensor. MFCCs describe the spectral envelope — vocal-tract shape and voice quality, which is what emotion changes.',
    detail: 'n_fft 2048 · hop 512 · 128 mels',
  },
  {
    icon: Layers,
    title: 'Augment (train only)',
    body: 'Two extra copies per clip: mild noise, time-stretch, pitch-shift and gain. Ranges stay small so the emotion label survives the transform.',
    detail: 'validation and test stay clean',
  },
  {
    icon: Boxes,
    title: 'Train & compare',
    body: 'A logistic-regression baseline, a CNN, a BiLSTM and a CNN-LSTM, all fed identical audio. Early stopping monitors validation macro-F1, not accuracy.',
    detail: 'class weights counter the neutral imbalance',
  },
  {
    icon: SlidersHorizontal,
    title: 'Select & calibrate',
    body: 'The simplest model within one bootstrap standard error of the best wins. A confidence threshold is swept on validation data so low-confidence answers can be flagged.',
    detail: 'test set never consulted',
  },
]

export default function Method({ dataset }) {
  const reduced = useReducedMotion()

  return (
    <section className="mx-auto max-w-6xl px-5 py-20 sm:py-24">
      <SectionHeading
        id="method"
        eyebrow="Method"
        title="The pipeline, end to end"
        description="Nothing here is decorative. Each step exists to stop a specific failure mode — leakage, overfitting, or a model that looks better than it is."
      />

      <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {STEPS.map((s, i) => (
          <motion.div
            key={s.title}
            initial={reduced ? false : { opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.45, delay: i * 0.07, ease: EASE }}
          >
            <Card className="flex h-full flex-col p-5">
              <div className="mb-3 flex items-center gap-2.5">
                <span className="flex size-8 items-center justify-center rounded-lg bg-surface-3">
                  <s.icon className="size-4 text-accent" aria-hidden="true" />
                </span>
                <span className="tnum text-xs font-medium text-faint">
                  0{i + 1}
                </span>
              </div>
              <h3 className="text-sm font-semibold text-fg">{s.title}</h3>
              <p className="mt-1.5 flex-1 text-[13px] leading-relaxed text-muted">
                {s.body}
              </p>
              <p className="mt-3 border-t border-line pt-2.5 text-[11px] uppercase tracking-wide text-faint">
                {s.detail}
              </p>
            </Card>
          </motion.div>
        ))}

        {/* split card */}
        <Reveal delay={0.35}>
          <Card className="flex h-full flex-col p-5">
            <div className="mb-3 flex items-center gap-2.5">
              <span className="flex size-8 items-center justify-center rounded-lg bg-primary">
                <GitBranch className="size-4 text-primary-fg" aria-hidden="true" />
              </span>
              <span className="text-xs font-semibold uppercase tracking-wide text-primary">
                the critical bit
              </span>
            </div>
            <h3 className="text-sm font-semibold text-fg">Actor-disjoint split</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-muted">
              A random file-level split would put the same actor in both train and
              test, letting the network recognise the <em>voice</em> instead of the
              emotion. Actors are partitioned instead, and the code asserts the
              sets never intersect.
            </p>
            <dl className="mt-3.5 space-y-1.5 border-t border-line pt-3">
              {[
                ['train', dataset.splitActors.train, dataset.splitFiles.train],
                ['val', dataset.splitActors.val, dataset.splitFiles.val],
                ['test', dataset.splitActors.test, dataset.splitFiles.test],
              ].map(([name, actors, n]) => (
                <div key={name} className="flex items-baseline gap-2 text-[11px]">
                  <dt className="w-10 shrink-0 font-semibold uppercase tracking-wide text-fg">
                    {name}
                  </dt>
                  <dd className="tnum truncate text-muted" title={actors.join(', ')}>
                    {actors.length} actors · {n} clips
                  </dd>
                </div>
              ))}
            </dl>
          </Card>
        </Reveal>
      </div>
    </section>
  )
}

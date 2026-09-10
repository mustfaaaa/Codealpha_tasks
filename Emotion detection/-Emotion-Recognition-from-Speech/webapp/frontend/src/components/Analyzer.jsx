import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle, CheckCircle2, FileAudio, Loader2, Play, Radio, Upload, X,
} from 'lucide-react'
import { audioUrl, getSamples, predictFile, predictSample } from '../lib/api'
import { emotionColor, emotionLabel } from '../lib/emotions'
import ConfidenceRing from './ConfidenceRing'
import ProbabilityBars from './ProbabilityBars'
import { Card, Chip, EASE, Reveal, SectionHeading } from './ui'

export default function Analyzer({ isDark, threshold }) {
  const reduced = useReducedMotion()
  const [samples, setSamples] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef(null)
  const audioRef = useRef(null)

  useEffect(() => {
    getSamples().then(setSamples).catch((e) => setError(e.message))
  }, [])

  const run = useCallback(async (fn, id) => {
    setLoading(true)
    setError(null)
    setActiveId(id)
    try {
      setResult(await fn())
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const onSample = (s, { autoplay = true } = {}) => {
    run(() => predictSample(s.id), s.id)
    if (audioRef.current) {
      audioRef.current.src = audioUrl(s.id)
      // Never autoplay on a deep link: sound starting by itself on page load
      // is hostile, and it is exactly the case the user did not opt into.
      if (autoplay) audioRef.current.play().catch(() => {})
    }
  }

  const onFile = (file) => {
    if (!file) return
    run(() => predictFile(file), file.name)
    if (audioRef.current) {
      audioRef.current.src = URL.createObjectURL(file)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    onFile(e.dataTransfer.files?.[0])
  }

  const testSamples = samples.filter((s) => s.kind === 'test')
  const degraded = samples.filter((s) => s.kind === 'degraded')

  return (
    <section className="mx-auto max-w-6xl px-5 py-20 sm:py-24">
      <SectionHeading
        id="analyze"
        eyebrow="Try it"
        title="Analyse a recording"
        description="Pick a clip from the held-out test speakers, or drop in your own audio. Every prediction below is produced live by the exported model — nothing is pre-computed."
      />

      <div className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* ------------------------------------------------------- input -- */}
        <Reveal className="space-y-5">
          {/* Dropzone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <motion.button
              type="button"
              onClick={() => fileInput.current?.click()}
              animate={{
                borderColor: dragging ? 'var(--accent)' : 'var(--line)',
                backgroundColor: dragging ? 'color-mix(in srgb, var(--accent) 8%, transparent)' : 'var(--surface-2)',
              }}
              whileHover={reduced ? {} : { y: -2 }}
              transition={{ duration: 0.2 }}
              className="flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center"
            >
              <motion.span
                animate={dragging && !reduced ? { y: [-2, 2, -2] } : { y: 0 }}
                transition={{ repeat: dragging ? Infinity : 0, duration: 1.4 }}
                className="mb-3 flex size-11 items-center justify-center rounded-full bg-surface-3"
              >
                <Upload className="size-5 text-accent" aria-hidden="true" />
              </motion.span>
              <span className="text-sm font-medium text-fg">
                Drop an audio file, or click to browse
              </span>
              <span className="mt-1 text-xs text-muted">
                WAV, MP3, FLAC, OGG or M4A · up to 20 MB
              </span>
            </motion.button>
            <input
              ref={fileInput}
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
              className="sr-only"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
          </div>

          {/* Held-out samples */}
          <div>
            <div className="mb-2.5 flex items-center gap-2">
              <Radio className="size-4 text-primary" aria-hidden="true" />
              <h3 className="text-sm font-semibold text-fg">
                Held-out test clips
              </h3>
              <span className="text-xs text-faint">actors 20–23, never trained on</span>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-2 xl:grid-cols-4">
              {testSamples.map((s) => {
                const active = activeId === s.id
                const color = emotionColor(s.label, isDark)
                return (
                  <motion.button
                    key={s.id}
                    type="button"
                    onClick={() => onSample(s)}
                    whileHover={reduced ? {} : { y: -2 }}
                    whileTap={reduced ? {} : { scale: 0.98 }}
                    transition={{ duration: 0.18 }}
                    className={`group relative cursor-pointer overflow-hidden rounded-lg border px-3 py-2.5 text-left transition-colors ${
                      active
                        ? 'border-transparent bg-surface-3'
                        : 'border-line bg-surface-2 hover:border-line-strong'
                    }`}
                    aria-pressed={active}
                  >
                    {active && (
                      <motion.span
                        layoutId="sample-active"
                        className="absolute inset-y-0 left-0 w-[3px]"
                        style={{ background: color }}
                        transition={{ duration: 0.3, ease: EASE }}
                      />
                    )}
                    <span className="flex items-center gap-1.5">
                      <span
                        className="size-2 shrink-0 rounded-full"
                        style={{ background: color }}
                        aria-hidden="true"
                      />
                      <span className="text-[13px] font-medium text-fg">
                        {emotionLabel(s.label)}
                      </span>
                    </span>
                    <span className="mt-0.5 block text-[11px] text-faint">
                      actor {s.actor} · {s.gender}
                    </span>
                  </motion.button>
                )
              })}
            </div>
          </div>

          {/* Degraded-channel samples */}
          {degraded.length > 0 && (
            <div>
              <div className="mb-2.5 flex items-center gap-2">
                <AlertTriangle className="size-4 text-warn" aria-hidden="true" />
                <h3 className="text-sm font-semibold text-fg">Degraded channel</h3>
              </div>
              <p className="mb-2.5 text-xs leading-relaxed text-muted">
                The same held-out clips, pushed through a telephone band or a
                reverberant room. These test robustness to an unseen recording
                channel — accuracy drops, and so should confidence.
              </p>
              <div className="flex flex-wrap gap-2">
                {degraded.map((s) => (
                  <motion.button
                    key={s.id}
                    type="button"
                    onClick={() => onSample(s)}
                    whileHover={reduced ? {} : { y: -1 }}
                    whileTap={reduced ? {} : { scale: 0.98 }}
                    className={`cursor-pointer rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                      activeId === s.id
                        ? 'border-warn text-fg'
                        : 'border-line bg-surface-2 text-muted hover:border-line-strong'
                    }`}
                  >
                    {emotionLabel(s.label)} · {s.intensity}
                  </motion.button>
                ))}
              </div>
            </div>
          )}

          <audio ref={audioRef} controls preload="none" className="w-full rounded-lg">
            <track kind="captions" />
          </audio>
        </Reveal>

        {/* ------------------------------------------------------ output -- */}
        <Reveal delay={0.08}>
          <Card className="min-h-[520px] overflow-hidden p-6">
            <AnimatePresence mode="wait">
              {loading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex h-[470px] flex-col items-center justify-center gap-3"
                >
                  <Loader2 className="size-7 animate-spin text-accent" aria-hidden="true" />
                  <p className="text-sm text-muted">Extracting MFCCs and running the network…</p>
                </motion.div>
              )}

              {!loading && error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="flex h-[470px] flex-col items-center justify-center gap-3 text-center"
                >
                  <span className="flex size-11 items-center justify-center rounded-full bg-[color-mix(in_srgb,var(--primary)_14%,transparent)]">
                    <X className="size-5 text-primary" aria-hidden="true" />
                  </span>
                  <p className="max-w-xs text-sm text-fg">{error}</p>
                  <p className="text-xs text-muted">
                    Check the backend is running on port 5001.
                  </p>
                </motion.div>
              )}

              {!loading && !error && !result && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex h-[470px] flex-col items-center justify-center gap-3 text-center"
                >
                  <FileAudio className="size-8 text-faint" aria-hidden="true" />
                  <p className="text-sm font-medium text-fg">No audio analysed yet</p>
                  <p className="max-w-xs text-xs leading-relaxed text-muted">
                    Choose one of the held-out clips or upload your own file to
                    see the full probability distribution across all eight emotions.
                  </p>
                </motion.div>
              )}

              {!loading && !error && result && (
                <motion.div
                  key={result.audio + result.predicted_emotion}
                  initial={reduced ? false : { opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={reduced ? {} : { opacity: 0, y: -8 }}
                  transition={{ duration: 0.35, ease: EASE }}
                >
                  <Result result={result} isDark={isDark} threshold={threshold} />
                </motion.div>
              )}
            </AnimatePresence>
          </Card>
        </Reveal>
      </div>
    </section>
  )
}

function Result({ result, isDark, threshold }) {
  const color = emotionColor(result.predicted_emotion, isDark)
  const correct = result.trueLabel
    ? result.trueLabel === result.predicted_emotion
    : null

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider text-faint">
            Predicted emotion
          </p>
          <motion.h3
            className="mt-1 text-3xl font-semibold tracking-tight"
            style={{ color }}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, ease: EASE }}
          >
            {emotionLabel(result.predicted_emotion)}
          </motion.h3>
          <p className="mt-1 truncate text-xs text-muted" title={result.audio}>
            {result.audio?.split(/[\\/]/).pop()}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {result.source && (
            <Chip>{result.source === 'test' ? 'held-out speaker'
              : result.source === 'degraded' ? 'degraded channel' : 'your upload'}</Chip>
          )}
          {correct !== null && (
            <Chip tone={correct ? 'ok' : 'warn'}>
              {correct
                ? <><CheckCircle2 className="size-3.5" aria-hidden="true" /> matches label</>
                : <><AlertTriangle className="size-3.5" aria-hidden="true" /> true: {emotionLabel(result.trueLabel)}</>}
            </Chip>
          )}
        </div>
      </div>

      <div className="mt-6 flex flex-col items-center gap-5 sm:flex-row sm:items-center sm:gap-7">
        <ConfidenceRing
          value={result.confidence}
          threshold={threshold ?? result.confidence_threshold}
          color={color}
          reliable={result.reliable}
        />
        <div className="flex-1">
          <p className="text-sm leading-relaxed text-muted">
            {result.reliable ? (
              <>The model clears its calibrated threshold of{' '}
                <span className="tnum font-medium text-fg">
                  {((threshold ?? result.confidence_threshold) * 100).toFixed(0)}%
                </span>
                , so this prediction is reported as confident. On the test split,
                predictions above this line were right{' '}
                <span className="tnum font-medium text-fg">77.8%</span> of the time.
              </>
            ) : (
              <>Confidence sits <em>below</em> the calibrated threshold of{' '}
                <span className="tnum font-medium text-fg">
                  {((threshold ?? result.confidence_threshold) * 100).toFixed(0)}%
                </span>
                . The system reports this as uncertain rather than asserting an
                answer it cannot support.
              </>
            )}
          </p>
        </div>
      </div>

      <div className="mt-6 border-t border-line pt-5">
        <h4 className="mb-3 text-sm font-semibold text-fg">
          Probability across all eight classes
        </h4>
        <ProbabilityBars
          probabilities={result.probabilities}
          isDark={isDark}
          predicted={result.predicted_emotion}
        />
      </div>
    </div>
  )
}

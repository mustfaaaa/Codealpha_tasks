import { AnimatePresence, motion, useReducedMotion, useScroll, useSpring } from 'framer-motion'
import { useEffect, useState } from 'react'
import { AudioLines, Loader2, Moon, Sun } from 'lucide-react'
import Analyzer from './components/Analyzer'
import Dashboard from './components/Dashboard'
import Hero from './components/Hero'
import Method from './components/Method'
import { getOverview } from './lib/api'
import { EASE } from './components/ui'

const NAV = [
  { href: '#analyze', label: 'Analyse' },
  { href: '#performance', label: 'Performance' },
  { href: '#method', label: 'Method' },
]

export default function App() {
  const reduced = useReducedMotion()
  const [theme, setTheme] = useState(
    () => localStorage.getItem('ser-theme') ?? 'dark',
  )
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const { scrollYProgress } = useScroll()
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 28, restDelta: 0.001 })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem('ser-theme', theme)
    } catch {
      /* private mode — the theme just won't persist */
    }
  }, [theme])

  useEffect(() => {
    getOverview().then(setData).catch((e) => setError(e.message))
  }, [])

  // The browser resolves #hash before React has rendered the sections, so a
  // shared deep link would otherwise land at the top of the page.
  useEffect(() => {
    if (!data || !window.location.hash) return
    const target = document.querySelector(window.location.hash)
    if (target) {
      requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }))
    }
  }, [data])

  const isDark = theme === 'dark'

  if (error) {
    return (
      <main className="flex min-h-dvh items-center justify-center px-6">
        <div className="max-w-md text-center">
          <h1 className="text-lg font-semibold text-fg">Cannot reach the model API</h1>
          <p className="mt-2 text-sm text-muted">{error}</p>
          <p className="mt-4 rounded-lg border border-line bg-surface-2 p-3 text-left text-xs text-muted">
            Start the backend first:
            <code className="mt-1.5 block font-mono text-fg">
              .venv\Scripts\python.exe webapp\backend\app.py
            </code>
          </p>
        </div>
      </main>
    )
  }

  if (!data) {
    return (
      <main className="flex min-h-dvh flex-col items-center justify-center gap-3">
        <Loader2 className="size-6 animate-spin text-accent" aria-hidden="true" />
        <p className="text-sm text-muted">Loading model metrics…</p>
      </main>
    )
  }

  return (
    <div className="min-h-dvh bg-surface">
      {/* scroll progress */}
      <motion.div
        className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left bg-primary"
        style={{ scaleX: reduced ? 1 : progress, opacity: reduced ? 0 : 1 }}
        aria-hidden="true"
      />

      <a
        href="#analyze"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-primary focus:px-4 focus:py-2 focus:text-sm focus:text-primary-fg"
      >
        Skip to the analyser
      </a>

      <Nav theme={theme} setTheme={setTheme} />

      <main>
        <Hero dataset={data.dataset} test={data.test} />
        <Analyzer isDark={isDark} threshold={data.model.threshold} />
        <Dashboard data={data} isDark={isDark} />
        <Method dataset={data.dataset} />
      </main>

      <Footer model={data.model} unseen={data.unseen} />
    </div>
  )
}

function Nav({ theme, setTheme }) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <motion.nav
      animate={{
        backgroundColor: scrolled
          ? 'color-mix(in srgb, var(--surface) 82%, transparent)'
          : 'transparent',
        borderColor: scrolled ? 'var(--line)' : 'transparent',
      }}
      transition={{ duration: 0.25 }}
      className="sticky top-0 z-40 border-b backdrop-blur-md"
    >
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
        <a href="#top" className="flex items-center gap-2">
          <span className="flex size-7 items-center justify-center rounded-md bg-primary">
            <AudioLines className="size-4 text-primary-fg" aria-hidden="true" />
          </span>
          <span className="text-sm font-semibold tracking-tight text-fg">
            SER Studio
          </span>
        </a>

        <div className="flex items-center gap-1">
          <ul className="mr-1 flex items-center gap-0.5 sm:gap-1">
            {NAV.map((n) => (
              <li key={n.href}>
                <a
                  href={n.href}
                  className="block rounded-lg px-2 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-surface-2 hover:text-fg sm:px-3 sm:text-[13px]"
                >
                  {n.label}
                </a>
              </li>
            ))}
          </ul>

          <button
            type="button"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            className="flex size-9 cursor-pointer items-center justify-center rounded-lg border border-line bg-surface-2 text-muted transition-colors hover:text-fg"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={theme}
                initial={{ opacity: 0, rotate: -70, scale: 0.7 }}
                animate={{ opacity: 1, rotate: 0, scale: 1 }}
                exit={{ opacity: 0, rotate: 70, scale: 0.7 }}
                transition={{ duration: 0.22, ease: EASE }}
                className="flex"
              >
                {theme === 'dark'
                  ? <Sun className="size-4" aria-hidden="true" />
                  : <Moon className="size-4" aria-hidden="true" />}
              </motion.span>
            </AnimatePresence>
          </button>
        </div>
      </div>
    </motion.nav>
  )
}

function Footer({ model, unseen }) {
  const sim = unseen.simulated_channel
  return (
    <footer className="border-t border-line bg-surface-2/40">
      <div className="mx-auto max-w-6xl px-5 py-12">
        <div className="grid gap-8 sm:grid-cols-3">
          <div>
            <h3 className="text-sm font-semibold text-fg">Honest limits</h3>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              RAVDESS is <em>acted</em> emotion — two fixed sentences, studio-recorded.
              Real spontaneous affect is subtler. Sixteen training actors is few,
              and per-speaker test accuracy spans 61.7% to 78.3%.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-fg">Channel mismatch</h3>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              On telephone-band and reverberant versions of the same clips,
              accuracy fell to{' '}
              <span className="tnum font-medium text-fg">
                {sim ? `${sim.n_correct}/${sim.n}` : 'n/a'}
              </span>
              . Usefully, confidence fell with it — the model became uncertain
              rather than confidently wrong.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-fg">Model</h3>
            <dl className="mt-2 space-y-1 text-xs text-muted">
              <div className="flex justify-between gap-2">
                <dt>architecture</dt>
                <dd className="font-medium text-fg">{model.architecture.toUpperCase()}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt>parameters</dt>
                <dd className="tnum font-medium text-fg">{model.params.toLocaleString()}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt>classes</dt>
                <dd className="tnum font-medium text-fg">{model.classes.length}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt>threshold</dt>
                <dd className="tnum font-medium text-fg">{model.threshold.toFixed(2)}</dd>
              </div>
            </dl>
          </div>
        </div>
        <p className="mt-10 border-t border-line pt-5 text-xs text-faint">
          Dataset: RAVDESS (Livingstone &amp; Russo, 2018), CC BY-NC-SA 4.0.
          All metrics shown are read from the pipeline's own output files.
        </p>
      </div>
    </footer>
  )
}

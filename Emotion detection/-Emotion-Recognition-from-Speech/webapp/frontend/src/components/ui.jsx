import { motion, useInView, useReducedMotion, useSpring, useTransform } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

/* ---------------------------------------------------------------- motion --
   One shared set of variants keeps timing consistent across the whole app,
   instead of every component inventing its own duration.
--------------------------------------------------------------------------- */
export const EASE = [0.22, 0.61, 0.36, 1]

export const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE } },
}

export const stagger = (each = 0.06) => ({
  hidden: {},
  show: { transition: { staggerChildren: each } },
})

/** Reveals children once when scrolled into view. */
export function Reveal({ children, className = '', delay = 0, amount = 0.2 }) {
  const reduced = useReducedMotion()
  if (reduced) return <div className={className}>{children}</div>
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount }}
      variants={{
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE, delay } },
      }}
    >
      {children}
    </motion.div>
  )
}

/** Number that counts up when it enters the viewport. */
export function CountUp({ value, decimals = 0, suffix = '', prefix = '', className = '' }) {
  const reduced = useReducedMotion()
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, amount: 0.5 })
  const spring = useSpring(0, { stiffness: 70, damping: 20 })
  const text = useTransform(spring, (v) => `${prefix}${v.toFixed(decimals)}${suffix}`)

  useEffect(() => {
    if (inView) spring.set(value)
  }, [inView, value, spring])

  if (reduced) {
    return (
      <span ref={ref} className={`tnum ${className}`}>
        {prefix}{value.toFixed(decimals)}{suffix}
      </span>
    )
  }
  return <motion.span ref={ref} className={`tnum ${className}`}>{text}</motion.span>
}

/* ----------------------------------------------------------------- shell -- */
export function Card({ children, className = '', ...rest }) {
  return (
    <div
      className={`rounded-xl border border-line bg-surface-2 ${className}`}
      style={{ boxShadow: 'var(--shadow)' }}
      {...rest}
    >
      {children}
    </div>
  )
}

export function SectionHeading({ eyebrow, title, description, id }) {
  return (
    <Reveal className="max-w-2xl">
      <div id={id} className="scroll-mt-24">
        {eyebrow && (
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
            {eyebrow}
          </p>
        )}
        <h2 className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">
          {title}
        </h2>
        {description && (
          <p className="mt-3 text-[15px] leading-relaxed text-muted">{description}</p>
        )}
      </div>
    </Reveal>
  )
}

export function Stat({ label, value, sub, accent = false }) {
  return (
    <Card className="p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-faint">{label}</p>
      <p
        className={`mt-1.5 text-2xl font-semibold tracking-tight ${
          accent ? 'text-primary' : 'text-fg'
        }`}
      >
        {value}
      </p>
      {sub && <p className="mt-1 text-xs leading-snug text-muted">{sub}</p>}
    </Card>
  )
}

export function Chip({ children, tone = 'neutral', className = '' }) {
  const tones = {
    neutral: 'border-line bg-surface-3 text-muted',
    accent: 'border-transparent bg-accent text-accent-fg',
    ok: 'border-transparent text-[var(--ok)] bg-[color-mix(in_srgb,var(--ok)_14%,transparent)]',
    warn: 'border-transparent text-[var(--warn)] bg-[color-mix(in_srgb,var(--warn)_16%,transparent)]',
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

/** Tooltip-on-focus/hover helper for terse metric labels. */
export function Hint({ children, text }) {
  const [open, setOpen] = useState(false)
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span tabIndex={0} className="cursor-help border-b border-dotted border-faint">
        {children}
      </span>
      {open && (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 z-30 mb-2 w-56 -translate-x-1/2 rounded-lg border border-line bg-surface px-3 py-2 text-xs font-normal leading-relaxed text-muted shadow-lg"
        >
          {text}
        </span>
      )}
    </span>
  )
}

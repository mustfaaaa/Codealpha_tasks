/**
 * Emotion identity: colour + icon glyph per class.
 *
 * Colour is never the only channel — every bar, cell and chip also carries a
 * text label and a number, so the UI stays readable for colour-blind users and
 * in greyscale print.
 */
export const EMOTION_ORDER = [
  'neutral', 'calm', 'happy', 'sad',
  'angry', 'fearful', 'disgust', 'surprised',
]

export const EMOTION_META = {
  neutral:   { color: '#64748b', dark: '#94a3b8', label: 'Neutral' },
  calm:      { color: '#0d9488', dark: '#2dd4bf', label: 'Calm' },
  happy:     { color: '#d97706', dark: '#fbbf24', label: 'Happy' },
  sad:       { color: '#2563eb', dark: '#60a5fa', label: 'Sad' },
  angry:     { color: '#dc2626', dark: '#f87171', label: 'Angry' },
  fearful:   { color: '#7c3aed', dark: '#a78bfa', label: 'Fearful' },
  disgust:   { color: '#15803d', dark: '#4ade80', label: 'Disgust' },
  surprised: { color: '#c026d3', dark: '#e879f9', label: 'Surprised' },
}

/** Arousal grouping — this is what the model's errors actually cluster by. */
export const AROUSAL = {
  high: ['angry', 'happy', 'surprised', 'fearful'],
  low: ['sad', 'calm', 'neutral', 'disgust'],
}

export function emotionColor(name, isDark) {
  const meta = EMOTION_META[name]
  if (!meta) return isDark ? '#94a3b8' : '#64748b'
  return isDark ? meta.dark : meta.color
}

export function emotionLabel(name) {
  return EMOTION_META[name]?.label ?? name
}

export const pct = (v, digits = 1) =>
  `${(v * 100).toFixed(digits)}%`

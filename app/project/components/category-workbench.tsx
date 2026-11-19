'use client'

import { useState } from 'react'
import { ArrowDown, ArrowRight, Bell, Check, ChevronDown, CircleHelp, Command, Copy, Filter, Inbox, MessageSquareText, MoreHorizontal, Search, Send, Settings2, Sparkles, ThumbsUp, X } from 'lucide-react'

type Result = { id: string; text: string; author: string; source: string; category: string; subject: string; score: number; memberships: string[]; action: string; conflict: string; time: string }

const initialResults: Result[] = [
  { id: 'CMT-2841', text: 'Love how lightweight this is. Been using it every day since launch!', author: 'Maya Chen', source: 'Instagram · @mayamakes', category: 'Appreciation', subject: 'Positive feedback on lightweight design', score: 96, memberships: ['Praise', 'Product feedback'], action: 'Thank them for sharing, and invite them to leave a review.', conflict: 'No conflicting signals detected.', time: '2m ago' },
  { id: 'CMT-2840', text: 'Does this come in a larger size? I couldn’t find the size chart anywhere.', author: 'Jordan Lee', source: 'Website · Product page', category: 'Query', subject: 'Asking about larger sizes and size chart', score: 93, memberships: ['Question', 'Purchase intent'], action: 'Share the size chart and confirm available sizes.', conflict: 'Question is primary; purchase intent is a supporting signal.', time: '8m ago' },
  { id: 'CMT-2839', text: 'first!! 🔥🔥', author: 'thevibeclub', source: 'TikTok · @northstar', category: 'Noise', subject: 'Low-context engagement', score: 88, memberships: ['Generic engagement'], action: 'No action needed. Keep it out of the support queue.', conflict: 'No conflicting signals detected.', time: '14m ago' },
  { id: 'CMT-2838', text: 'Third time reaching out and my order still hasn’t arrived. Really disappointed.', author: 'Sam Rivera', source: 'Instagram · @northstar', category: 'Bad experience', subject: 'Delayed order after multiple support attempts', score: 98, memberships: ['Complaint', 'Delivery', 'Urgent'], action: 'Escalate to support and check the shipment status. Reply publicly to acknowledge.', conflict: 'Negative experience overrides the implied order-status question.', time: '21m ago' },
  { id: 'CMT-2837', text: 'Would you ever make a version for sensitive skin?', author: 'Ari Williams', source: 'TikTok · @northstar', category: 'Query', subject: 'Asking about a sensitive-skin formula', score: 91, memberships: ['Question', 'Product feedback'], action: 'Answer with current ingredients and log the feature request.', conflict: 'Product suggestion is secondary to the direct question.', time: '36m ago' },
]
const categories = ['All comments', 'Query', 'Noise', 'Appreciation', 'Bad experience']
const tone: Record<string, string> = { Query: 'query', Noise: 'noise', Appreciation: 'appreciation', 'Bad experience': 'bad' }

type ClassifierResponse = {
  item_id: string
  primary_category: string
  memberships: Record<string, number>
  conflict_resolution: string
  actionability_score: number
  recommended_action: string
}

type Analysis = Omit<Result, 'id' | 'text' | 'author' | 'source' | 'time'>

const categoryByKey: Record<string, string> = { real_query: 'Query', noise: 'Noise', appreciation: 'Appreciation', bad_experience: 'Bad experience' }
const subjectByCategory: Record<string, string> = {
  Query: 'Customer question or product inquiry',
  Noise: 'Low-context engagement',
  Appreciation: 'Positive feedback and customer appreciation',
  'Bad experience': 'Customer reporting an issue that needs attention',
}
const MEMBERSHIP_THRESHOLD = 0.2

function toAnalysis(data: ClassifierResponse): Analysis {
  const category = categoryByKey[data.primary_category] ?? 'Query'
  return {
    category,
    subject: subjectByCategory[category],
    score: Math.round(data.actionability_score),
    memberships: Object.entries(data.memberships)
      .filter(([, membership]) => membership >= MEMBERSHIP_THRESHOLD)
      .map(([key]) => categoryByKey[key] ?? key),
    action: data.recommended_action,
    conflict: data.conflict_resolution,
  }
}

export function CategoryWorkbench() {
  const [items, setItems] = useState(initialResults)
  const [filter, setFilter] = useState('All comments')
  const [message, setMessage] = useState('')
  const [selected, setSelected] = useState<Result | null>(null)
  const [notice, setNotice] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const visible = filter === 'All comments' ? items : items.filter((item) => item.category === filter)

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const text = message.trim()
    if (!text || analyzing) return
    setAnalyzing(true)
    setError('')
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
      if (!response.ok) throw new Error('Analysis failed')
      const data = (await response.json()) as ClassifierResponse
      setItems([{ ...toAnalysis(data), id: data.item_id, text, author: 'You · New comment', source: 'Manual entry', time: 'just now' }, ...items])
      setMessage('')
      setFilter('All comments')
    } catch {
      setError('Classifier unavailable. Make sure it is running on localhost.')
    } finally {
      setAnalyzing(false)
    }
  }

  return <main className="workspace-shell">
    <aside className="sidebar">
      <a className="brand" href="#home"><span className="brand-mark"><MessageSquareText size={19} /></span><span>signal<span className="brand-dot">.</span></span></a>
      <div className="workspace-switch"><div className="workspace-avatar">N</div><div className="workspace-label"><strong>Northstar Studio</strong><span>Free workspace</span></div><ChevronDown size={15} /></div>
      <div className="side-caption">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        <a className="nav-link active" href="#inbox"><Inbox size={17} /> Inbox <span className="nav-count">12</span></a>
        <a className="nav-link" href="#insights"><span className="nav-icon"><Sparkles size={17} /></span> Insights</a>
      </nav>
      <div className="side-caption channels-caption">CHANNELS <button aria-label="Add channel">+</button></div>
      <div className="channel-list"><div><span className="channel-dot instagram">◎</span> Instagram <span className="channel-count">8</span></div><div><span className="channel-dot tiktok">♪</span> TikTok <span className="channel-count">3</span></div><div><span className="channel-dot web">◉</span> Website <span className="channel-count">1</span></div></div>
      <div className="sidebar-bottom"><div className="plan-card"><div className="plan-top"><span>Monthly usage</span><span>68%</span></div><div className="usage-track"><span /></div><p>682 of 1,000 comments</p><button>Upgrade plan <ArrowRight size={13} /></button></div><a className="nav-link" href="#settings"><Settings2 size={17} /> Settings</a><a className="profile" href="#profile"><div className="profile-avatar">JD</div><div><strong>Jamie Davis</strong><span>Owner</span></div><MoreHorizontal size={17} /></a></div>
    </aside>

    <section className="main-panel">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>Inbox</strong></div><div className="top-actions"><button className="icon-button" aria-label="Search"><Search size={17} /></button><button className="icon-button notification-button" aria-label="Notifications"><Bell size={17} /><i /></button><div className="top-divider" /><button className="help-button"><CircleHelp size={16} /> Help</button><div className="avatar-small">JD</div></div></header>
      <div className="page-content">
        <div className="page-heading"><div><div className="eyebrow"><span className="live-dot" /> YOUR SOCIAL INBOX</div><h1>Comments, understood.</h1><p>Every comment, sorted into a next step.</p></div><button className="date-button">Last 7 days <ChevronDown size={14} /></button></div>
        <section className="metrics" aria-label="Inbox overview"><div className="metric-card"><div className="metric-label">Total comments <span className="metric-icon purple"><MessageSquareText size={15} /></span></div><div className="metric-bottom"><strong>1,284</strong><span className="metric-trend">↑ 12.8%</span></div><div className="metric-foot">vs. previous 7 days</div></div><div className="metric-card"><div className="metric-label">Need a reply <span className="metric-icon amber"><ArrowRight size={15} /></span></div><div className="metric-bottom"><strong>24</strong><span className="metric-trend muted-trend">6 urgent</span></div><div className="metric-foot">across all channels</div></div><div className="metric-card"><div className="metric-label">Positive sentiment <span className="metric-icon green"><ThumbsUp size={15} /></span></div><div className="metric-bottom"><strong>72<span className="percent">%</span></strong><span className="metric-trend">↑ 4.2%</span></div><div className="metric-foot">of comments this week</div></div><div className="metric-card"><div className="metric-label">Avg. actionability <span className="metric-icon blue"><Sparkles size={15} /></span></div><div className="metric-bottom"><strong>86<span className="out-of">/100</span></strong><span className="metric-trend">↑ 2.1%</span></div><div className="metric-foot">classification confidence</div></div></section>

        <section className="analyzer"><div className="analyzer-icon"><Sparkles size={18} /></div><div className="analyzer-content"><div className="analyzer-title"><h2>Try the comment analyzer</h2><span>AI-POWERED</span></div><p>Paste any comment. Get its category, context, and what to do next.</p><form onSubmit={submit} className="analyzer-form"><label htmlFor="comment-input" className="sr-only">Paste a comment to analyze</label><textarea id="comment-input" value={message} onChange={(e) => setMessage(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && e.metaKey && !e.nativeEvent.isComposing && e.keyCode !== 229) { e.preventDefault(); e.currentTarget.form?.requestSubmit() } }} placeholder="Paste a comment here… e.g. ‘Love the product, but shipping took forever.’" rows={2} /><div className="analyzer-footer"><span role={error ? 'alert' : undefined}>{error || <><Command size={12} /> Press ⌘ Enter to analyze</>}</span><button type="submit" disabled={!message.trim() || analyzing}><Sparkles size={14} /> {analyzing ? 'Analyzing…' : 'Analyze comment'}</button></div></form></div></section>

        <div className="inbox-heading"><div><h2>Recent comments</h2><span className="result-total">{visible.length} comments</span></div><button className="filter-button"><Filter size={15} /> Filters <span>2</span></button></div>
        <div className="filters" role="tablist" aria-label="Filter comments">{categories.map((category) => <button key={category} role="tab" aria-selected={filter === category} className={filter === category ? 'filter-tab selected' : 'filter-tab'} onClick={() => setFilter(category)}>{category}{category === 'All comments' && <span className="tab-count">12</span>}</button>)}</div>
        <div className="comments-list">{visible.map((item) => <article className="comment-row" key={item.id}><div className={`source-avatar ${item.source.startsWith('Instagram') ? 'ig' : item.source.startsWith('TikTok') ? 'tt' : item.source.startsWith('Website') ? 'web-source' : 'manual'}`}>{item.source.startsWith('Instagram') ? '◎' : item.source.startsWith('TikTok') ? '♪' : item.source.startsWith('Website') ? '◉' : '✳'}</div><div className="comment-main"><div className="comment-meta"><strong>{item.author}</strong><span className="meta-dot">·</span><span>{item.source}</span><span className="meta-dot">·</span><span>{item.time}</span></div><p className="comment-text">{item.text}</p><div className="comment-summary"><span className={`category-pill ${tone[item.category]}`}>{item.category}</span><span className="subject-text">{item.subject}</span></div></div><div className="comment-right"><span className="score-label">{item.score}<span> / 100</span></span><button className="preview-button" onClick={() => setSelected(item)}>Preview <ArrowRight size={13} /></button></div></article>)}{visible.length === 0 && <div className="empty-list">No comments in this category yet.</div>}</div>
        <div className="list-footer"><span>Showing {visible.length} of 1,284 comments</span><button>View all comments <ArrowRight size={13} /></button></div>
        <footer className="page-footer"><span>Made for the conversations that move you forward.</span><span>Signal v1.0 <span className="footer-dot">·</span> All systems operational <i /></span></footer>
      </div>
    </section>

    {selected && <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) setSelected(null) }}><section className="preview-modal" role="dialog" aria-modal="true" aria-labelledby="preview-title"><div className="modal-header"><div><div className="eyebrow">COMMENT PREVIEW <span>·</span> {selected.id}</div><h2 id="preview-title">Classification details</h2></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="Close preview"><X size={18} /></button></div><blockquote>“{selected.text}”</blockquote><div className="preview-class"><span className={`category-pill ${tone[selected.category]}`}>{selected.category}</span><span className="score-label">Actionability <strong>{selected.score}/100</strong></span></div><div className="detail-block"><span className="detail-label">SUBJECT SUMMARY</span><p>{selected.subject}</p></div><div className="detail-block"><span className="detail-label">MEMBERSHIPS</span><div className="membership-list">{selected.memberships.map((m) => <span key={m}>{m}</span>)}</div></div><div className="detail-block"><span className="detail-label">CONFLICT NOTE</span><p>{selected.conflict}</p></div><div className="recommended"><span className="detail-label"><Sparkles size={12} /> RECOMMENDED ACTION</span><p>{selected.action}</p></div><div className="modal-actions"><button className="secondary-action" onClick={() => { navigator.clipboard?.writeText(selected.action); setNotice('Action copied'); setTimeout(() => setNotice(''), 1800) }}><Copy size={14} /> {notice || 'Copy action'}</button><button className="primary-action" onClick={() => setSelected(null)}><Check size={15} /> Mark reviewed</button></div></section></div>}
  </main>
}

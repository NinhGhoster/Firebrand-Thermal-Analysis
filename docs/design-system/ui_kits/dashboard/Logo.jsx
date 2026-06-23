// Logo.jsx — wordmark with inferno-gradient mark.
function Logo({ size = 'sm' }) {
  const dims = size === 'lg'
    ? { sq: 40, t: 22, sub: 11 }
    : { sq: 24, t: 14, sub: 9 };
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{
        width: dims.sq, height: dims.sq, borderRadius: 6,
        background: 'var(--gradient-inferno)',
        boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.1)',
        position: 'relative',
        overflow: 'hidden',
        flexShrink: 0,
      }}>
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon name="flame" size={Math.round(dims.sq * 0.65)} color="rgba(0,0,0,0.55)" />
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1 }}>
        <div style={{
          font: `700 ${dims.t}px/1 var(--font-ui)`,
          color: 'var(--fg-primary)',
          letterSpacing: '-0.01em',
        }}>Firebrand</div>
        <div style={{
          font: `500 ${dims.sub}px/1 var(--font-data)`,
          color: 'var(--fg-muted)',
          marginTop: 4,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          whiteSpace: 'nowrap',
        }}>Thermal Analysis</div>
      </div>
    </div>
  );
}

window.Logo = Logo;

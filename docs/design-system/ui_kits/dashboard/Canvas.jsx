// Canvas.jsx — the central thermal viewport.
// Uses the captured screenshot as the still-frame backdrop, with overlays drawn on top.
function Canvas({ colormap, roi, frame, zoom, isPlaying, onHover, hoverTemp }) {
  // Inferno is the original colormap of the source frame, so no filter for that.
  // Other colormaps approximate by hue-rotating the inferno frame.
  const colorTints = {
    Inferno:   'none',
    Jet:       'hue-rotate(180deg) saturate(1.2)',
    Hot:       'saturate(1.4) brightness(1.05)',
    Magma:     'hue-rotate(-15deg) saturate(0.9)',
    Plasma:    'hue-rotate(-40deg) saturate(1.1)',
    Bone:      'grayscale(1) sepia(0.15) brightness(0.95)',
    Turbo:     'hue-rotate(140deg) saturate(1.5)',
    Grayscale: 'grayscale(1)',
  };

  return (
    <div className="canvas"
         onMouseMove={(e) => {
           const rect = e.currentTarget.getBoundingClientRect();
           onHover && onHover({ x: e.clientX - rect.left, y: e.clientY - rect.top });
         }}
         onMouseLeave={() => onHover && onHover(null)}>

      {/* Thermal frame backdrop — uses the captured project still */}
      <div className="canvas__frame" style={{ filter: colorTints[colormap] || 'none' }} />

      {/* Subtle grain / film texture so the frame doesn't look static */}
      <div className="canvas__grain" />

      {/* ROI rectangle (green, 2px stroke, preserved from source) */}
      {roi && (
        <div className="canvas__roi" style={{
          left: `${roi.x}%`, top: `${roi.y}%`,
          width: `${roi.w}%`, height: `${roi.h}%`,
        }}>
          <div className="canvas__roi-tag">ROI · {Math.round(roi.x*10.24)},{Math.round(roi.y*7.68)},{Math.round(roi.w*10.24)},{Math.round(roi.h*7.68)}</div>
        </div>
      )}

      {/* Tracking marker — a tiny dot + label, moves with frame */}
      <div className="canvas__track" style={{
        left: `${48 + Math.sin(frame * 0.04) * 6}%`,
        top:  `${56 + Math.cos(frame * 0.05) * 4}%`,
      }}>
        <div className="canvas__track-dot" />
        <div className="canvas__track-label">Tracking · 306.9°C</div>
      </div>

      {/* Hover temperature tooltip */}
      {hoverTemp && (
        <div className="canvas__tooltip" style={{ left: hoverTemp.x + 12, top: hoverTemp.y + 12 }}>
          <span className="canvas__tooltip-temp">{hoverTemp.temp.toFixed(1)}°C</span>
          <span className="canvas__tooltip-coord">{hoverTemp.px},{hoverTemp.py}</span>
        </div>
      )}

      {/* Zoom indicator (only when zoomed) */}
      {zoom !== 100 && (
        <div className="canvas__zoom">{zoom}%</div>
      )}

      {/* Frame counter overlay (top-left corner) */}
      <div className="canvas__frame-counter">
        <span style={{ color: 'var(--fg-muted)' }}>frame</span>{' '}
        <span style={{ color: 'var(--fg-primary)' }}>{String(frame).padStart(5, '0')}</span>
        <span style={{ color: 'var(--fg-muted)' }}> / 17318</span>
      </div>

      {/* REC dot when playing */}
      {isPlaying && (
        <div className="canvas__rec">
          <div className="canvas__rec-dot" />
          <span>LIVE</span>
        </div>
      )}
    </div>
  );
}

window.Canvas = Canvas;

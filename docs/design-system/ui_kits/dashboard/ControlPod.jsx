// ControlPod.jsx — bottom-pinned transport bar.
// Row 1: full-bleed scrub slider.
// Row 2: transport buttons (left) · status line (right).
function ControlPod({ frame, totalFrames, isPlaying, onPlay, onPrev, onNext, onStop, onScrub, statusText, statusTone = 'warn' }) {
  const pct = totalFrames ? (frame / totalFrames) * 100 : 0;

  const handleSliderClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    onScrub && onScrub(Math.round(x * totalFrames));
  };

  return (
    <div className="control-pod">
      {/* Row 1: scrub slider */}
      <div className="scrub" onClick={handleSliderClick}>
        <div className="scrub__track" />
        <div className="scrub__fill" style={{ width: `${pct}%` }} />
        <div className="scrub__handle" style={{ left: `${pct}%` }} />
        {/* tick markers every 10% */}
        {Array.from({ length: 11 }).map((_, i) => (
          <div key={i} className="scrub__tick" style={{ left: `${i * 10}%` }} />
        ))}
      </div>

      {/* Row 2: transport + status */}
      <div className="transport">
        <div className="transport__left">
          <Button kind="primary" icon={isPlaying ? 'pause' : 'play'} onClick={onPlay} style={{ width: 84 }}>
            {isPlaying ? 'Pause' : 'Play'}
          </Button>
          <Button icon="chevron-left" onClick={onPrev} title="Previous frame" />
          <Button icon="chevron-right" onClick={onNext} title="Next frame" />
          <Button icon="square" onClick={onStop} title="Stop" />
        </div>

        <div className={`status status--${statusTone}`}>{statusText}</div>
      </div>
    </div>
  );
}

window.ControlPod = ControlPod;

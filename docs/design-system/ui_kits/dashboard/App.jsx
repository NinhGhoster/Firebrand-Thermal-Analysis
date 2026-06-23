// App.jsx — top-level composition.

const { useState, useEffect, useRef, useCallback } = React;

const SCENE_PRESETS = [
  { file: 'Rec-0158.seq', total: 17318, emiss: 0.940 },
  { file: 'Rec-0211.csq', total: 9842,  emiss: 0.920 },
  { file: 'Burn-04-W.nc', total: 24108, emiss: 0.940 },
];

function App() {
  const [colormap, setColormap] = useState('Inferno');
  const [frame, setFrame] = useState(8854);
  const [isPlaying, setIsPlaying] = useState(false);
  const [fileIndex, setFileIndex] = useState(0);
  const [zoom, setZoom] = useState(100);
  const [hover, setHover] = useState(null);
  const [thresh, setThresh] = useState('300.0');
  const [emiss, setEmiss] = useState('0.940');
  const [startF, setStartF] = useState('1');
  const [endF, setEndF] = useState('max');
  const [roi, setRoi] = useState({ x: 1, y: 1, w: 99, h: 44 });
  const [status, setStatus] = useState({ tone: 'warn', text: 'Status: ready' });
  const playRef = useRef();

  const scene = SCENE_PRESETS[fileIndex];

  // Drive frame counter when playing
  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(() => {
      setFrame((f) => {
        const next = f + 1;
        if (next >= scene.total) { setIsPlaying(false); return scene.total; }
        return next;
      });
    }, 1000 / 30);
    playRef.current = id;
    return () => clearInterval(id);
  }, [isPlaying, scene.total]);

  // Keyboard: space toggles play/pause
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') { e.preventDefault(); setIsPlaying((p) => !p); }
      if (e.code === 'ArrowRight') setFrame((f) => Math.min(f + 1, scene.total));
      if (e.code === 'ArrowLeft')  setFrame((f) => Math.max(f - 1, 0));
      if (e.key && /^[1-8]$/.test(e.key)) {
        const cmaps = ['Inferno','Jet','Hot','Magma','Plasma','Bone','Turbo','Grayscale'];
        setColormap(cmaps[parseInt(e.key) - 1]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [scene.total]);

  const flash = useCallback((text, tone = 'warn', dur = 2200) => {
    setStatus({ text, tone });
    clearTimeout(flash._t);
    flash._t = setTimeout(() => {
      setStatus({ tone: 'warn', text: `Status: frame ${frame}/${scene.total} | ROI: (${roi.x*10}, ${roi.y*8}, ${roi.w*10}, ${roi.h*8}) | thresh: ${thresh}C` });
    }, dur);
  }, [frame, roi, thresh, scene.total]);

  // Live status when not flashing
  useEffect(() => {
    if (status.tone === 'warn' && !status.text.includes('opening') && !status.text.includes('exported')) {
      setStatus({ tone: 'warn', text: `Status: frame ${frame}/${scene.total} | ROI: (${roi.x*10}, ${roi.y*8}, ${roi.w*10}, ${roi.h*8}) | thresh: ${thresh}C` });
    }
  }, [frame, roi, thresh, fileIndex]);

  // Hover -> fake temperature value
  const onCanvasHover = useCallback((pt) => {
    if (!pt) return setHover(null);
    // pretend the hot core is near 60% across, 75% down
    const dx = (pt.x / 800) - 0.5;
    const dy = (pt.y / 460) - 0.7;
    const dist = Math.hypot(dx, dy);
    const temp = Math.max(28, 720 * Math.exp(-dist * 3.5) + Math.random() * 4);
    setHover({ x: pt.x, y: pt.y, temp, px: Math.round(pt.x * 1.28), py: Math.round(pt.y * 1.67) });
  }, []);

  return (
    <div className="app">
      <div className="app__stage">
        <div className="app__main">
          <Canvas
            colormap={colormap}
            roi={roi}
            frame={frame}
            zoom={zoom}
            isPlaying={isPlaying}
            onHover={onCanvasHover}
            hoverTemp={hover}
          />
          <ColorBar colormap={colormap} tmin={0} tmax={795} />
        </div>

        <SidePanel
          filename={scene.file}
          fileIndex={fileIndex + 1}
          fileCount={SCENE_PRESETS.length}
          colormap={colormap}
          onColormap={(c) => { setColormap(c); flash(`Status: colormap → ${c}`, 'warn'); }}
          zoom={zoom}
          onResetZoom={() => setZoom(100)}
          thresh={thresh}
          emiss={emiss}
          emissMeta={scene.emiss.toFixed(3)}
          startF={startF}
          endF={endF}
          roi={roi}
          currentFrame={frame}
          fileName={scene.file}
          onOpen={() => flash('Status: opened ' + scene.file + ' | 1024×768 | ' + scene.total + ' frames')}
          onPrev={() => { setFileIndex((i) => (i - 1 + SCENE_PRESETS.length) % SCENE_PRESETS.length); setFrame(1); flash('Status: switched to previous file'); }}
          onNext={() => { setFileIndex((i) => (i + 1) % SCENE_PRESETS.length); setFrame(1); flash('Status: switched to next file'); }}
          onApplyFile={() => flash('Status: settings applied to ' + scene.file, 'ok')}
          onApplyAll={() => flash('Status: settings applied to all ' + SCENE_PRESETS.length + ' files', 'ok')}
          onAutoRoi={() => { setRoi({ x: 1, y: 1, w: 99, h: 44 }); flash('Status: auto ROI set to (3, 7, 1020, 373)', 'ok'); }}
          onResetRoi={() => { setRoi({ x: 0, y: 0, w: 100, h: 100 }); flash('Status: ROI cleared'); }}
          onExport={() => flash('Status: exporting ' + scene.file.replace(/\.\w+$/, '.csv') + ' · 24,108 rows', 'ok', 3000)}
          onCheckUpdates={() => flash('Status: you are up to date (v0.0.3)', 'info')}
        />
      </div>

      <ControlPod
        frame={frame}
        totalFrames={scene.total}
        isPlaying={isPlaying}
        onPlay={() => setIsPlaying((p) => !p)}
        onPrev={() => setFrame((f) => Math.max(0, f - 1))}
        onNext={() => setFrame((f) => Math.min(scene.total, f + 1))}
        onStop={() => { setIsPlaying(false); setFrame(0); }}
        onScrub={(f) => setFrame(f)}
        statusText={status.text}
        statusTone={status.tone}
      />
    </div>
  );
}

window.App = App;

// SidePanel.jsx — right-side scrollable stack of pods.
// Order matches the source app: Data Source / Visualisation / Parameters / Exports / Footer.

const COLORMAPS = ['Inferno','Jet','Hot','Magma','Plasma','Bone','Turbo','Grayscale'];

function DataSourcePod({ filename, fileIndex, fileCount, onOpen, onPrev, onNext }) {
  return (
    <Pod title="Data Source">
      <Button kind="primary" icon="folder-open" onClick={onOpen} style={{ width: '100%', justifyContent: 'center' }}>
        Open File/Folder
      </Button>
      <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
        <Button icon="chevrons-left" onClick={onPrev} style={{ flex: 1, justifyContent: 'center' }}>Prev</Button>
        <Button icon="chevrons-right" onClick={onNext} style={{ flex: 1, justifyContent: 'center' }}>
          <span>Next</span>
        </Button>
      </div>
      {filename && (
        <div className="pod__meta" style={{ marginTop: 10 }}>
          <div className="pod__meta-key">file</div>
          <div className="pod__meta-val mono" title={filename}>{filename}</div>
          <div className="pod__meta-key">index</div>
          <div className="pod__meta-val mono">{fileIndex} / {fileCount}</div>
        </div>
      )}
    </Pod>
  );
}

function VisualisationPod({ colormap, onColormap, zoom, onResetZoom }) {
  return (
    <Pod title="Visualisation">
      <div className="cmap-list">
        {COLORMAPS.map((name, i) => {
          const active = name === colormap;
          return (
            <div key={name}
                 className={`cmap-row ${active ? 'cmap-row--active' : ''}`}
                 onClick={() => onColormap(name)}>
              <span className="cmap-row__key">{i + 1}</span>
              <span className="cmap-row__name">{name}</span>
              <div className="cmap-row__ramp" data-cmap={name} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
        <Button icon="rotate-ccw" onClick={onResetZoom} style={{ flex: 1, justifyContent: 'center' }}>
          Reset Zoom
        </Button>
        <span className="mono" style={{ minWidth: 48, textAlign: 'right', color: 'var(--fg-muted)', fontSize: 12 }}>{zoom}%</span>
      </div>
    </Pod>
  );
}

function ParametersPod({ thresh, emiss, emissMeta, startF, endF, roi, onApplyFile, onApplyAll, onAutoRoi, onResetRoi, currentFrame, fileName }) {
  return (
    <Pod title="Parameters">
      <Field label="Detect Thresh">
        <Input value={thresh} width={92} />
        <span className="unit">°C</span>
      </Field>
      <Field label="Emissivity">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--fg-muted)' }}>{emissMeta}</span>
          <Input value={emiss} width={68} />
        </div>
      </Field>

      <div className="field-group">
        <div className="field-group__title">Export Range</div>
        <div className="range-row">
          <span className="range-row__lbl">Start</span>
          <Input value={startF} width={64} />
          <Button size="sm">Set · {currentFrame}</Button>
        </div>
        <div className="range-row">
          <span className="range-row__lbl">End</span>
          <Input value={endF} width={64} />
          <Button size="sm">Set · {currentFrame}</Button>
        </div>
      </div>

      <div className="field-group">
        <div className="field-group__title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>Region of Interest</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--fg-muted)' }}>x  y  w  h</span>
        </div>
        <div className="roi-row">
          <Input value={roi.x} width="100%" />
          <Input value={roi.y} width="100%" />
          <Input value={roi.w} width="100%" />
          <Input value={roi.h} width="100%" />
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
          <Button icon="crosshair" onClick={onAutoRoi} style={{ flex: 1, justifyContent: 'center' }}>Auto Target Fuel</Button>
          <Button onClick={onResetRoi}>Reset</Button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
        <Button onClick={onApplyFile} style={{ flex: 1, justifyContent: 'center' }} title={`Apply to ${fileName}`}>Apply file</Button>
        <Button onClick={onApplyAll} style={{ flex: 1, justifyContent: 'center' }}>Apply all</Button>
      </div>
    </Pod>
  );
}

function ExportPod({ onExport }) {
  return (
    <Pod title="Export">
      <Button kind="primary" icon="download" onClick={onExport} style={{ width: '100%', justifyContent: 'center', height: 40 }}>
        Export Actions
      </Button>
      <div className="export-options">
        <div className="export-options__item">
          <span>CSV (current file)</span>
          <span className="mono kbd">⌘E</span>
        </div>
        <div className="export-options__item">
          <span>CSV (all files · parallel)</span>
          <span className="mono kbd">⌘⇧E</span>
        </div>
        <div className="export-options__item">
          <span>Frame image (JPG)</span>
          <span className="mono kbd">⌘J</span>
        </div>
      </div>
    </Pod>
  );
}

function FooterPod({ onCheckUpdates }) {
  return (
    <Pod>
      <div className="footer-credit">
        <div className="footer-credit__line">Developed by</div>
        <div className="footer-credit__authors">H. Nguyen · J. Filippi · T. Penman · M. Peace · A. Filkov</div>
        <div className="footer-credit__version mono">v0.0.3 · build 240414</div>
      </div>
      <Button onClick={onCheckUpdates} icon="refresh" style={{ width: '100%', justifyContent: 'center' }}>
        Check for updates
      </Button>
    </Pod>
  );
}

function SidePanel(props) {
  return (
    <aside className="side-panel">
      <div className="side-panel__brand">
        <Logo size="sm" />
        <Button icon="settings" size="sm" title="Preferences" />
      </div>

      <DataSourcePod {...props} />
      <VisualisationPod {...props} />
      <ParametersPod {...props} />
      <ExportPod {...props} />
      <FooterPod {...props} />
    </aside>
  );
}

window.SidePanel = SidePanel;

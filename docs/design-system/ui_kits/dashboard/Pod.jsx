// Pod.jsx — standard side-panel card with ember section header.
// Matches CTkFrame(corner_radius=10) in the source app.
function Pod({ title, children, action }) {
  return (
    <div className="pod">
      {(title || action) && (
        <div className="pod__head">
          {title && <div className="pod__title">{title}</div>}
          {action}
        </div>
      )}
      <div className="pod__body">{children}</div>
    </div>
  );
}

// Field.jsx — labelled row inside a pod.
function Field({ label, hint, children }) {
  return (
    <div className="field">
      <div className="field__label">{label}</div>
      <div className="field__control">{children}</div>
      {hint && <div className="field__hint">{hint}</div>}
    </div>
  );
}

window.Pod = Pod;
window.Field = Field;

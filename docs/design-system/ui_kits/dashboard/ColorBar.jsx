// ColorBar.jsx — vertical temperature legend, pinned to the right of the canvas.
function ColorBar({ colormap, tmin = 0, tmax = 795 }) {
  // map colormap name -> CSS gradient
  const ramps = {
    Inferno:   'linear-gradient(180deg,#FCFFA4,#F7D03C,#FB9A06,#ED6925,#CF4446,#A52C60,#781C6D,#4A0C6B,#1B0C42,#000004)',
    Jet:       'linear-gradient(180deg,#7F0000,#FF0000,#FFFF00,#00FFFF,#0020FF,#00008F)',
    Hot:       'linear-gradient(180deg,#FFFFFF,#FFFF00,#FF7F00,#FF0000,#7F0000,#000000)',
    Magma:     'linear-gradient(180deg,#FCFDBF,#FDD9A0,#FB9C7C,#F0746E,#CF4148,#A6225C,#771C6D,#451A6B,#180F3D,#000004)',
    Plasma:    'linear-gradient(180deg,#F0F921,#FCCE25,#FCA636,#F1844B,#E16462,#CB4779,#B12A90,#900DA4,#6A00A8,#42039D,#0D0887)',
    Bone:      'linear-gradient(180deg,#FFFFFF,#B7BFC9,#80808E,#56566D,#2B2B3C,#000000)',
    Turbo:     'linear-gradient(180deg,#7A0403,#C8260C,#FF7900,#F5BD24,#A8F03B,#41F087,#1AD0CA,#3F9EF7,#4146A6,#30123B)',
    Grayscale: 'linear-gradient(180deg,#FFFFFF,#000000)',
  };
  return (
    <div className="colorbar">
      <div className="colorbar__strip" style={{ background: ramps[colormap] || ramps.Inferno }}>
        <div className="colorbar__label colorbar__label--top">{tmax}°</div>
        <div className="colorbar__label colorbar__label--bot">{tmin}°</div>
      </div>
      <div className="colorbar__unit">°C</div>
    </div>
  );
}

window.ColorBar = ColorBar;

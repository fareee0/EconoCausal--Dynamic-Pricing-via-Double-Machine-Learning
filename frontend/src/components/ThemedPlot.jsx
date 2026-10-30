import Plot from 'react-plotly.js';

const chartColors = ['#55d4c5', '#b6a2ff', '#f282bd', '#82adff', '#f3c46f'];

const axisTheme = {
  color: '#d1cbed',
  gridcolor: 'rgba(190, 177, 255, 0.16)',
  linecolor: 'rgba(190, 177, 255, 0.24)',
  zerolinecolor: 'rgba(190, 177, 255, 0.26)',
};

export default function ThemedPlot({ layout = {}, ...props }) {
  const { font, legend, xaxis, yaxis, ...layoutOptions } = layout;

  return (
    <Plot
      {...props}
      layout={{
        ...layoutOptions,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        colorway: chartColors,
        font: {
          family: 'Inter, sans-serif',
          color: '#eeebff',
          size: 12,
          ...font,
        },
        hoverlabel: {
          bgcolor: '#17142f',
          bordercolor: 'rgba(190, 177, 255, 0.45)',
          font: { color: '#f7f5ff' },
        },
        legend: {
          ...legend,
          font: { color: '#e8e3ff', ...legend?.font },
        },
        xaxis: { ...xaxis, ...axisTheme },
        yaxis: { ...yaxis, ...axisTheme },
      }}
    />
  );
}

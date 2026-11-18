import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface PosteriorAreaChartProps {
  gamma: number[][]; // [T, 3]
  stateNames?: string[];
  activeStep?: number;
  onStepChange?: (step: number) => void;
  height?: number;
  title?: string;
}

export const PosteriorAreaChart: React.FC<PosteriorAreaChartProps> = ({
  gamma,
  stateNames = ['Walking', 'Sitting', 'Running'],
  activeStep = 0,
  onStepChange,
  height = 200,
  title = 'Posterior State Probability Distribution γ_t(i)',
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !gamma || gamma.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const containerWidth = svgRef.current.parentElement?.clientWidth || 650;
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };
    const width = containerWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const T = gamma.length;
    const colors = ['#5B8266', '#C98A2C', '#C4603A'];

    const g = svg
      .attr('width', containerWidth)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3
      .scaleLinear()
      .domain([0, T - 1])
      .range([0, width]);

    const yScale = d3.scaleLinear().domain([0, 1.0]).range([innerHeight, 0]);

    // Layer 0: Walking [0 -> gamma[t][0]]
    const areaWalking = d3
      .area<number[]>()
      .x((_, idx) => xScale(idx))
      .y0(() => yScale(0))
      .y1((d) => yScale(d[0]))
      .curve(d3.curveMonotoneX);

    // Layer 1: Sitting [gamma[t][0] -> gamma[t][0] + gamma[t][1]]
    const areaSitting = d3
      .area<number[]>()
      .x((_, idx) => xScale(idx))
      .y0((d) => yScale(d[0]))
      .y1((d) => yScale(d[0] + d[1]))
      .curve(d3.curveMonotoneX);

    // Layer 2: Running [gamma[t][0] + gamma[t][1] -> 1.0]
    const areaRunning = d3
      .area<number[]>()
      .x((_, idx) => xScale(idx))
      .y0((d) => yScale(d[0] + d[1]))
      .y1((d) => yScale(Math.min(d[0] + d[1] + d[2], 1.0)))
      .curve(d3.curveMonotoneX);

    // Render layers
    g.append('path')
      .datum(gamma)
      .attr('d', areaWalking)
      .attr('fill', colors[0])
      .attr('opacity', 0.85);

    g.append('path')
      .datum(gamma)
      .attr('d', areaSitting)
      .attr('fill', colors[1])
      .attr('opacity', 0.85);

    g.append('path')
      .datum(gamma)
      .attr('d', areaRunning)
      .attr('fill', colors[2])
      .attr('opacity', 0.85);

    // Y Axis
    g.append('g')
      .call(
        d3
          .axisLeft(yScale)
          .ticks(5)
          .tickFormat((d: any) => `${Number(d).toFixed(1)}`)
      )
      .call((axis) => axis.select('.domain').attr('stroke', '#CBB99F'))
      .call((axis) => axis.selectAll('.tick line').attr('stroke', '#E2D3BC'))
      .call((axis) =>
        axis
          .selectAll('.tick text')
          .attr('fill', '#6B5B4E')
          .attr('font-size', '10px')
          .attr('font-family', 'var(--font-mono)')
      );

    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .ticks(Math.min(T, 12))
          .tickFormat((d: any) => `${d}`)
      )
      .call((axis) => axis.select('.domain').attr('stroke', '#CBB99F'))
      .call((axis) => axis.selectAll('.tick line').attr('stroke', '#E2D3BC'))
      .call((axis) =>
        axis
          .selectAll('.tick text')
          .attr('fill', '#6B5B4E')
          .attr('font-size', '10px')
          .attr('font-family', 'var(--font-mono)')
      );

    // Active Step cursor
    if (activeStep >= 0 && activeStep < T) {
      g.append('line')
        .attr('x1', xScale(activeStep))
        .attr('x2', xScale(activeStep))
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', '#2E2118')
        .attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '3,3');
    }

    // Transparent overlay for scrubbing interaction
    const overlay = g
      .append('rect')
      .attr('width', width)
      .attr('height', innerHeight)
      .attr('fill', 'transparent')
      .attr('cursor', 'crosshair');

    const handleMouse = (event: MouseEvent) => {
      if (!onStepChange) return;
      const [mx] = d3.pointer(event);
      const stepIdx = Math.round(xScale.invert(mx));
      const clamped = Math.max(0, Math.min(stepIdx, T - 1));
      onStepChange(clamped);
    };

    overlay.on('click', handleMouse);
    overlay.on('mousemove', (event) => {
      if (event.buttons === 1) handleMouse(event);
    });
  }, [gamma, stateNames, activeStep, onStepChange, height]);

  const currentGamma = gamma && gamma[activeStep] ? gamma[activeStep] : [0, 0, 0];

  return (
    <div
      style={{
        backgroundColor: 'var(--surface-card)',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-sm)',
        padding: '14px 16px',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '8px',
        }}
      >
        <span
          style={{
            fontSize: '0.88rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-serif)',
          }}
        >
          {title}
        </span>

        {/* Live instantaneous probabilities */}
        <div style={{ display: 'flex', gap: '12px', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
          <span style={{ color: '#5B8266' }}>
            Walking: <strong>{(currentGamma[0] * 100).toFixed(1)}%</strong>
          </span>
          <span style={{ color: '#C98A2C' }}>
            Sitting: <strong>{(currentGamma[1] * 100).toFixed(1)}%</strong>
          </span>
          <span style={{ color: '#C4603A' }}>
            Running: <strong>{(currentGamma[2] * 100).toFixed(1)}%</strong>
          </span>
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg ref={svgRef} />
      </div>
    </div>
  );
};

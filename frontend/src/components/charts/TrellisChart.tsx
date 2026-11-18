import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface TrellisChartProps {
  probabilities: number[][]; // [T, N] - e.g. alpha_scaled or beta_scaled or gamma
  stateNames?: string[];
  activeStep: number;
  onSelectStep?: (step: number) => void;
  height?: number;
  label?: string;
}

export const TrellisChart: React.FC<TrellisChartProps> = ({
  probabilities,
  stateNames = ['Walking', 'Sitting', 'Running'],
  activeStep,
  onSelectStep,
  height = 240,
  label = 'Inference Lattice Trellis',
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !probabilities || probabilities.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const containerWidth = svgRef.current.parentElement?.clientWidth || 700;
    const margin = { top: 25, right: 30, bottom: 35, left: 80 };
    const width = containerWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const T = probabilities.length;
    const N = stateNames.length;

    const stateColors = ['#5B8266', '#C98A2C', '#C4603A'];

    const g = svg
      .attr('width', containerWidth)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain([0, Math.max(T - 1, 1)])
      .range([0, width]);

    const yScale = d3
      .scalePoint<string>()
      .domain(stateNames)
      .range([0, innerHeight])
      .padding(0.4);

    // Grid lines for time steps
    g.append('g')
      .attr('class', 'grid-x')
      .selectAll('line')
      .data(xScale.ticks(Math.min(T, 15)))
      .enter()
      .append('line')
      .attr('x1', (d) => xScale(d))
      .attr('x2', (d) => xScale(d))
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .attr('stroke', '#EFE4D2')
      .attr('stroke-dasharray', '2,2');

    // Transitions lines between consecutive time steps
    if (T <= 60) {
      for (let t = 0; t < T - 1; t++) {
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) {
            const prob1 = probabilities?.[t]?.[i] ?? 0;
            const prob2 = probabilities?.[t + 1]?.[j] ?? 0;
            const weight = prob1 * prob2;
            if (weight > 0.005) {
              g.append('line')
                .attr('x1', xScale(t))
                .attr('y1', yScale(stateNames[i]) || 0)
                .attr('x2', xScale(t + 1))
                .attr('y2', yScale(stateNames[j]) || 0)
                .attr('stroke', stateColors[i])
                .attr('stroke-width', Math.max(weight * 3.5, 0.5))
                .attr('stroke-opacity', Math.min(Math.max(weight * 1.5, 0.15), 0.75));
            }
          }
        }
      }
    }

    // Y Axis (State labels)
    g.append('g')
      .call(d3.axisLeft(yScale))
      .call((axis) => axis.select('.domain').attr('stroke', '#CBB99F'))
      .call((axis) => axis.selectAll('.tick line').attr('stroke', '#E2D3BC'))
      .call((axis) =>
        axis
          .selectAll('.tick text')
          .attr('fill', '#2E2118')
          .attr('font-size', '12px')
          .attr('font-weight', '600')
          .attr('font-family', 'var(--font-sans)')
      );

    // X Axis (Time steps)
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .ticks(Math.min(T, 15))
          .tickFormat((d: any) => `${d}`)
      )
      .call((axis) => axis.select('.domain').attr('stroke', '#CBB99F'))
      .call((axis) => axis.selectAll('.tick line').attr('stroke', '#E2D3BC'))
      .call((axis) =>
        axis
          .selectAll('.tick text')
          .attr('fill', '#6B5B4E')
          .attr('font-size', '11px')
          .attr('font-family', 'var(--font-mono)')
      );

    // Active Step highlight vertical bar
    if (activeStep >= 0 && activeStep < T) {
      g.append('line')
        .attr('x1', xScale(activeStep))
        .attr('x2', xScale(activeStep))
        .attr('y1', -10)
        .attr('y2', innerHeight + 10)
        .attr('stroke', '#C4603A')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4,2')
        .attr('opacity', 0.85);

      g.append('text')
        .attr('x', xScale(activeStep))
        .attr('y', -12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#C4603A')
        .attr('font-size', '10px')
        .attr('font-weight', '700')
        .attr('font-family', 'var(--font-mono)')
        .text(`t = ${activeStep}`);
    }

    // Nodes (State circles at each step)
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < N; i++) {
        const p = probabilities?.[t]?.[i] ?? 0;
        const cx = xScale(t);
        const cy = yScale(stateNames[i]) || 0;
        const isActive = t === activeStep;

        const node = g
          .append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', Math.max(p * 9, 3))
          .attr('fill', stateColors[i])
          .attr('stroke', isActive ? '#2E2118' : '#FFFCF7')
          .attr('stroke-width', isActive ? 2 : 1)
          .attr('cursor', 'pointer')
          .attr('opacity', Math.max(p, 0.2));

        node.on('click', () => {
          if (onSelectStep) onSelectStep(t);
        });

        node.append('title').text(`t=${t}, ${stateNames[i]}: ${p.toFixed(5)}`);
      }
    }
  }, [probabilities, stateNames, activeStep, height, onSelectStep]);

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
          {label}
        </span>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          Click any node to scrub to that time step
        </span>
      </div>
      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg ref={svgRef} />
      </div>
    </div>
  );
};

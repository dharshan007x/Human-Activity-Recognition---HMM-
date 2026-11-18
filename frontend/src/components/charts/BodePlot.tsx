import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface BodePlotProps {
  frequencies: number[];
  magnitudeDb: number[];
  cutoffHz?: number;
  height?: number;
}

export const BodePlot: React.FC<BodePlotProps> = ({
  frequencies,
  magnitudeDb,
  cutoffHz = 10,
  height = 220,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !frequencies || frequencies.length === 0 || !magnitudeDb || magnitudeDb.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const containerWidth = svgRef.current.parentElement?.clientWidth || 550;
    const margin = { top: 20, right: 30, bottom: 35, left: 50 };
    const width = containerWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .attr('width', containerWidth)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const maxFreq = Math.max(...frequencies, 50);
    const xScale = d3.scaleLinear().domain([0, maxFreq]).range([0, width]);
    const yScale = d3.scaleLinear().domain([-60, 5]).range([innerHeight, 0]);

    // Horizontal grid
    g.append('g')
      .call(
        d3
          .axisLeft(yScale)
          .ticks(5)
          .tickSize(-width)
          .tickFormat(() => '')
      )
      .call((axis) => axis.select('.domain').remove())
      .call((axis) =>
        axis.selectAll('.tick line').attr('stroke', '#EFE4D2').attr('stroke-dasharray', '2,2')
      );

    // -3 dB cutoff horizontal threshold line
    const yCutoff = yScale(-3);
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yCutoff)
      .attr('y2', yCutoff)
      .attr('stroke', '#C98A2C')
      .attr('stroke-width', 1.2)
      .attr('stroke-dasharray', '3,3');

    // 10 Hz cutoff vertical line
    const xCutoff = xScale(cutoffHz);
    g.append('line')
      .attr('x1', xCutoff)
      .attr('x2', xCutoff)
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .attr('stroke', '#C4603A')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3');

    // Magnitude Line Generator
    const lineGen = d3
      .line<{ f: number; mag: number }>()
      .x((d) => xScale(d.f))
      .y((d) => yScale(d.mag))
      .curve(d3.curveMonotoneX);

    const data = frequencies.map((f, i) => ({ f, mag: magnitudeDb[i] }));

    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#366D67')
      .attr('stroke-width', 2.2)
      .attr('d', lineGen);

    // Left Axis (dB)
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => `${d} dB`))
      .call((axis) => axis.select('.domain').attr('stroke', '#CBB99F'))
      .call((axis) => axis.selectAll('.tick line').attr('stroke', '#E2D3BC'))
      .call((axis) =>
        axis
          .selectAll('.tick text')
          .attr('fill', '#6B5B4E')
          .attr('font-size', '10px')
          .attr('font-family', 'var(--font-mono)')
      );

    // Bottom Axis (Frequency Hz)
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(8).tickFormat((d) => `${d} Hz`))
      .call((axis) => axis.select('.domain').attr('stroke', '#CBB99F'))
      .call((axis) => axis.selectAll('.tick line').attr('stroke', '#E2D3BC'))
      .call((axis) =>
        axis
          .selectAll('.tick text')
          .attr('fill', '#6B5B4E')
          .attr('font-size', '10px')
          .attr('font-family', 'var(--font-mono)')
      );

    // Annotations
    g.append('text')
      .attr('x', xCutoff + 6)
      .attr('y', 14)
      .attr('fill', '#C4603A')
      .attr('font-size', '10px')
      .attr('font-weight', '700')
      .attr('font-family', 'var(--font-mono)')
      .text(`f_c = ${cutoffHz} Hz (-3dB)`);
  }, [frequencies, magnitudeDb, cutoffHz, height]);

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
          4th-Order Butterworth Frequency Response
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Anti-aliasing cutoff prior to M=4 decimation
        </span>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg ref={svgRef} />
      </div>
    </div>
  );
};

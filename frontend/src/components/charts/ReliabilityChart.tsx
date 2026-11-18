import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { CalibrationBin } from '../../services/api';

interface ReliabilityChartProps {
  bins: CalibrationBin[];
  ece?: number;
  height?: number;
}

export const ReliabilityChart: React.FC<ReliabilityChartProps> = ({
  bins,
  ece = 0.0,
  height = 240,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || !bins || bins.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const containerWidth = svgRef.current.parentElement?.clientWidth || 400;
    const margin = { top: 20, right: 25, bottom: 40, left: 45 };
    const width = containerWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .attr('width', containerWidth)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 1.0]).range([0, width]);
    const yScale = d3.scaleLinear().domain([0, 1.0]).range([innerHeight, 0]);

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
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

    // Diagonal perfect calibration line (y = x)
    g.append('line')
      .attr('x1', xScale(0))
      .attr('y1', yScale(0))
      .attr('x2', xScale(1))
      .attr('y2', yScale(1))
      .attr('stroke', '#CBB99F')
      .attr('stroke-width', 1.8)
      .attr('stroke-dasharray', '4,3');

    // Calibration bars / points
    const barWidth = width / 12;
    bins.forEach((bin) => {
      if (bin.count > 0) {
        // Bin bar
        g.append('rect')
          .attr('x', xScale(bin.confidence) - barWidth / 2)
          .attr('y', yScale(bin.accuracy))
          .attr('width', barWidth)
          .attr('height', innerHeight - yScale(bin.accuracy))
          .attr('fill', '#C4603A')
          .attr('opacity', 0.65)
          .attr('rx', 2);

        // Dot at top
        g.append('circle')
          .attr('cx', xScale(bin.confidence))
          .attr('cy', yScale(bin.accuracy))
          .attr('r', 3.5)
          .attr('fill', '#AA4E2B');
      }
    });

    // Y Axis (Empirical Accuracy)
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

    // X Axis (Mean Confidence)
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(
        d3
          .axisBottom(xScale)
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

    // Axis Labels
    g.append('text')
      .attr('x', width / 2)
      .attr('y', innerHeight + 32)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6B5B4E')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .text('Mean Predicted Confidence');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -32)
      .attr('text-anchor', 'middle')
      .attr('fill', '#6B5B4E')
      .attr('font-size', '11px')
      .attr('font-weight', '600')
      .text('Empirical Accuracy');
  }, [bins, height]);

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
          Model Reliability Diagram (Calibration)
        </span>

        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78rem',
            backgroundColor: 'var(--bg-secondary)',
            padding: '2px 8px',
            borderRadius: 'var(--radius-xs)',
            border: '1px solid var(--border-sand)',
            color: 'var(--accent-terracotta)',
            fontWeight: 700,
          }}
        >
          ECE = {ece.toFixed(4)}
        </span>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg ref={svgRef} />
      </div>
    </div>
  );
};

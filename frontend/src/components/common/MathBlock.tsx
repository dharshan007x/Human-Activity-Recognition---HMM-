import React, { useMemo } from 'react';
import katex from 'katex';

interface MathBlockProps {
  math: string;
  block?: boolean;
  className?: string;
}

export const MathBlock: React.FC<MathBlockProps> = ({ math, block = false, className = '' }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: block,
        throwOnError: false,
      });
    } catch {
      return math;
    }
  }, [math, block]);

  return (
    <span
      className={`katex-render ${block ? 'block my-3 overflow-x-auto py-1 text-center' : 'inline'} ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

import React from 'react';

export interface Column<T> {
  header: string;
  accessor?: keyof T;
  render?: (row: T, index: number) => React.ReactNode;
  align?: 'left' | 'center' | 'right';
  width?: string;
}

interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor?: (row: T, index: number) => string | number;
  emptyMessage?: string;
  maxHeight?: string;
}

export function Table<T>({
  columns,
  data,
  keyExtractor,
  emptyMessage = 'No records found.',
  maxHeight,
}: TableProps<T>) {
  return (
    <div
      style={{
        width: '100%',
        overflowX: 'auto',
        maxHeight: maxHeight || 'auto',
        border: '1px solid var(--border-sand)',
        borderRadius: 'var(--radius-sm)',
        backgroundColor: 'var(--surface-card)',
      }}
    >
      <table
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.86rem',
          textAlign: 'left',
        }}
      >
        <thead>
          <tr
            style={{
              backgroundColor: 'var(--bg-secondary)',
              borderBottom: '1.5px solid var(--border-sand)',
              position: 'sticky',
              top: 0,
              zIndex: 1,
            }}
          >
            {columns.map((col, idx) => (
              <th
                key={idx}
                style={{
                  padding: '10px 14px',
                  fontWeight: 600,
                  color: 'var(--text-secondary)',
                  textAlign: col.align || 'left',
                  width: col.width || 'auto',
                  textTransform: 'uppercase',
                  fontSize: '0.74rem',
                  letterSpacing: '0.04em',
                }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                style={{
                  padding: '28px',
                  textAlign: 'center',
                  color: 'var(--text-muted)',
                  fontStyle: 'italic',
                }}
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((row, rIdx) => (
              <tr
                key={keyExtractor ? keyExtractor(row, rIdx) : rIdx}
                style={{
                  borderBottom: '1px solid var(--border-light)',
                  backgroundColor: rIdx % 2 === 1 ? 'rgba(244, 235, 217, 0.25)' : 'transparent',
                  transition: 'background-color 0.15s ease',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-secondary)')}
                onMouseLeave={(e) =>
                  (e.currentTarget.style.backgroundColor =
                    rIdx % 2 === 1 ? 'rgba(244, 235, 217, 0.25)' : 'transparent')
                }
              >
                {columns.map((col, cIdx) => (
                  <td
                    key={cIdx}
                    style={{
                      padding: '9px 14px',
                      color: 'var(--text-primary)',
                      textAlign: col.align || 'left',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {col.render
                      ? col.render(row, rIdx)
                      : col.accessor
                      ? String(row[col.accessor] ?? '')
                      : ''}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

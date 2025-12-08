// web/next-demo/app/components/GraphCanvas.tsx
"use client";

import React, { useMemo } from "react";
import type { NodeId } from "./ControlsPanel";

export interface Edge {
  source: NodeId;
  target: NodeId;
  weight?: number;
  directed?: boolean;
}

export interface GraphCanvasProps {
  /**
   * Node identifiers in the graph.
   */
  nodes: NodeId[];

  /**
   * Edges between nodes.
   */
  edges: Edge[];

  /**
   * Nodes currently active (action = 1).
   */
  activeNodes?: NodeId[];

  /**
   * Nodes that belong to the forcing set, highlighted visually.
   */
  forcingSet?: NodeId[];

  /**
   * Optional nodes to emphasize even more, for example most influential.
   */
  highlightNodes?: NodeId[];

  /**
   * Pixel width and height of the SVG viewport.
   */
  width?: number;
  height?: number;

  /**
   * Optional label shown above the canvas.
   */
  title?: string;

  /**
   * Optional caption shown below the canvas.
   */
  caption?: string;
}

interface NodePosition {
  id: NodeId;
  x: number;
  y: number;
}

/**
 * Assign nodes to positions on a circle.
 */
function computeCircularLayout(
  nodes: NodeId[],
  width: number,
  height: number,
  padding: number
): NodePosition[] {
  if (nodes.length === 0) {
    return [];
  }

  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(
    10,
    Math.min(width, height) / 2 - padding
  );

  return nodes.map((id, index) => {
    const angle = (2 * Math.PI * index) / nodes.length;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    return { id, x, y };
  });
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  nodes,
  edges,
  activeNodes = [],
  forcingSet = [],
  highlightNodes = [],
  width = 480,
  height = 360,
  title,
  caption,
}) => {
  const nodeIds = useMemo(
    () => Array.from(new Set(nodes)),
    [nodes]
  );

  const activeSet = useMemo(
    () => new Set(activeNodes),
    [activeNodes]
  );
  const forcingSetIds = useMemo(
    () => new Set(forcingSet),
    [forcingSet]
  );
  const highlightSet = useMemo(
    () => new Set(highlightNodes),
    [highlightNodes]
  );

  const positions = useMemo(
    () => computeCircularLayout(nodeIds, width, height, 40),
    [nodeIds, width, height]
  );

  const positionsById = useMemo(() => {
    const map = new Map<NodeId, NodePosition>();
    for (const pos of positions) {
      map.set(pos.id, pos);
    }
    return map;
  }, [positions]);

  return (
    <section className="flex flex-col gap-2">
      {title && (
        <header>
          <h2 className="text-sm font-semibold text-gray-900">
            {title}
          </h2>
        </header>
      )}

      <div
        className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm"
        style={{ width, maxWidth: "100%" }}
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          className="block w-full h-auto"
        >
          {/* Edges */}
          <g stroke="#CBD5F5" strokeWidth={1.2}>
            {edges.map((edge, index) => {
              const from = positionsById.get(edge.source);
              const to = positionsById.get(edge.target);
              if (!from || !to) {
                return null;
              }

              const strokeWidth =
                edge.weight && edge.weight > 1
                  ? Math.min(2.5, 1 + edge.weight * 0.2)
                  : 1.2;

              return (
                <line
                  key={`${edge.source}-${edge.target}-${index}`}
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  strokeWidth={strokeWidth}
                  stroke="#CBD5F5"
                />
              );
            })}
          </g>

          {/* Nodes */}
          <g>
            {positions.map((pos) => {
              const isActive = activeSet.has(pos.id);
              const isForced = forcingSetIds.has(pos.id);
              const isHighlighted = highlightSet.has(pos.id);

              const baseRadius = 15;
              const radius = isHighlighted
                ? baseRadius + 4
                : baseRadius;

              const fillColor = isActive ? "#4F46E5" : "#F9FAFB";
              const strokeColor = isActive ? "#312E81" : "#9CA3AF";
              const strokeWidth = isForced ? 3 : 1.5;

              return (
                <g key={pos.id}>
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={radius}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                  />
                  <text
                    x={pos.x}
                    y={pos.y}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={11}
                    fill={isActive ? "#EEF2FF" : "#111827"}
                    style={{ userSelect: "none" }}
                  >
                    {pos.id}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {caption && (
        <p className="text-xs text-gray-600">{caption}</p>
      )}
    </section>
  );
};

export default GraphCanvas;
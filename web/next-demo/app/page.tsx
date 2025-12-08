// web/next-demo/app/page.tsx
"use client";

import React, { useCallback, useMemo, useState } from "react";

import ControlsPanel, {
  ControlsState,
  ExampleOption,
  InitialProfileMode,
} from "./components/ControlsPanel";
import type { NodeId } from "./components/ControlsPanel";
import GraphCanvas, { Edge } from "./components/GraphCanvas";

interface ExampleConfig {
  key: string;
  name: string;
  description?: string;
  nodes: NodeId[];
  edges: Edge[];
  defaultActiveNodes: NodeId[];
  defaultForcingSet: NodeId[];
}

/**
 * Static example catalog for the Next demo.
 *
 * This is a lightweight mirror of the Python examples so that you can
 * at least show something in the Next app, even though the actual
 * simulation engine lives in Python.
 */
const EXAMPLES: ExampleConfig[] = [
  {
    key: "triangle",
    name: "Triangle (3 nodes)",
    description:
      "3 node symmetric triangle with two PSNE in the full model: all 0 and all 1.",
    nodes: ["0", "1", "2"],
    edges: [
      { source: "0", target: "1" },
      { source: "1", target: "2" },
      { source: "2", target: "0" },
    ],
    defaultActiveNodes: [],
    defaultForcingSet: [],
  },
  {
    key: "dense_zealot",
    name: "Dense 80 percent with zealot",
    description:
      "Complete graph with one zealot and many high threshold nodes, inspired by the board example.",
    nodes: ["Z", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
    edges: (() => {
      const nodes: NodeId[] = [
        "Z",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
      ];
      const list: Edge[] = [];
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          list.push({ source: nodes[i], target: nodes[j], weight: 1 });
        }
      }
      return list;
    })(),
    defaultActiveNodes: ["Z"],
    defaultForcingSet: ["Z"],
  },
  {
    key: "line",
    name: "Line (5 nodes)",
    description:
      "Simple path graph. Good for showing contagion along a line.",
    nodes: ["0", "1", "2", "3", "4"],
    edges: [
      { source: "0", target: "1" },
      { source: "1", target: "2" },
      { source: "2", target: "3" },
      { source: "3", target: "4" },
    ],
    defaultActiveNodes: ["0"],
    defaultForcingSet: [],
  },
  {
    key: "random_network",
    name: "Random network (toy)",
    description:
      "Toy random graph. In the full system this would map to a random threshold game.",
    nodes: ["v0", "v1", "v2", "v3", "v4", "v5"],
    edges: [
      { source: "v0", target: "v1" },
      { source: "v0", target: "v2" },
      { source: "v1", target: "v3" },
      { source: "v2", target: "v4" },
      { source: "v3", target: "v5" },
      { source: "v4", target: "v5" },
    ],
    defaultActiveNodes: [],
    defaultForcingSet: [],
  },
];

function findExampleByKey(key: string): ExampleConfig | null {
  return EXAMPLES.find((ex) => ex.key === key) ?? null;
}

/**
 * Translate the initial profile mode and user selections into
 * the active node set we should display in the canvas.
 *
 * In this Next demo we are not running actual best response dynamics
 * in TypeScript. We simply show:
 *   - the example defaults, or
 *   - all inactive, or
 *   - the user chosen custom active set.
 */
function computeDisplayedActiveNodes(
  example: ExampleConfig,
  state: ControlsState
): NodeId[] {
  const mode: InitialProfileMode = state.initialMode;

  if (mode === "all_inactive") {
    return [];
  }

  if (mode === "custom") {
    return state.customActiveNodes;
  }

  return example.defaultActiveNodes;
}

export default function HomePage() {
  const [controlsState, setControlsState] = useState<ControlsState | null>(
    null
  );

  const examplesForPanel: ExampleOption[] = useMemo(
    () =>
      EXAMPLES.map((ex) => ({
        key: ex.key,
        name: ex.name,
        description: ex.description,
      })),
    []
  );

  const selectedExample: ExampleConfig | null = useMemo(() => {
    if (controlsState) {
      const withKey = findExampleByKey(controlsState.selectedExampleKey);
      if (withKey) {
        return withKey;
      }
    }
    return EXAMPLES[0] ?? null;
  }, [controlsState]);

  const displayedActiveNodes: NodeId[] = useMemo(() => {
    if (!selectedExample || !controlsState) {
      return selectedExample?.defaultActiveNodes ?? [];
    }
    return computeDisplayedActiveNodes(selectedExample, controlsState);
  }, [selectedExample, controlsState]);

  const displayedForcingSet: NodeId[] = useMemo(() => {
    if (!selectedExample || !controlsState) {
      return selectedExample?.defaultForcingSet ?? [];
    }
    return controlsState.forcingSet;
  }, [selectedExample, controlsState]);

  const handleControlsChange = useCallback((nextState: ControlsState) => {
    setControlsState(nextState);
  }, []);

  const handleRunCascade = useCallback((nextState: ControlsState) => {
    // In a full integration this would call an API that runs
    // the Python simulation and returns the final profile and
    // cascade history. For now we just treat this as "lock in"
    // the current settings so the canvas reflects them.
    setControlsState(nextState);
  }, []);

  const nodesForCanvas: NodeId[] = selectedExample
    ? selectedExample.nodes
    : [];
  const edgesForCanvas: Edge[] = selectedExample
    ? selectedExample.edges
    : [];

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 lg:flex-row">
        <div className="w-full lg:w-72 flex-shrink-0">
          <ControlsPanel
            examples={examplesForPanel}
            nodes={nodesForCanvas}
            defaultSelectedExampleKey={EXAMPLES[0]?.key}
            defaultForcingSet={EXAMPLES[0]?.defaultForcingSet}
            defaultInitialMode="default"
            defaultCustomActiveNodes={EXAMPLES[0]?.defaultActiveNodes}
            defaultTargetProfile="all_active"
            defaultMaxSteps={10}
            onRunCascade={handleRunCascade}
            onChange={handleControlsChange}
          />
        </div>

        <div className="flex flex-1 flex-col gap-4">
          <header className="flex flex-col gap-1">
            <h1 className="text-lg font-semibold text-gray-900">
              Influence game visual sandbox
            </h1>
            <p className="text-sm text-gray-600">
              This Next demo visualizes example graphs and your chosen
              forcing sets and initial profiles. The actual cascade and
              PSNE computation lives in the Python Streamlit app.
            </p>
          </header>

          <section className="flex flex-col gap-4">
            <GraphCanvas
              nodes={nodesForCanvas}
              edges={edgesForCanvas}
              activeNodes={displayedActiveNodes}
              forcingSet={displayedForcingSet}
              highlightNodes={displayedForcingSet}
              width={640}
              height={420}
              title={
                selectedExample?.name ??
                "Example influence game"
              }
              caption={
                selectedExample?.description ??
                "Example graph from the project."
              }
            />

            {controlsState && (
              <div className="rounded-md border border-gray-200 bg-white p-3 text-xs text-gray-800 shadow-sm">
                <h2 className="mb-1 text-sm font-semibold text-gray-900">
                  Current configuration
                </h2>
                <p>
                  <span className="font-medium">Example:</span>{" "}
                  {controlsState.selectedExampleKey}
                </p>
                <p>
                  <span className="font-medium">Forcing set:</span>{" "}
                  {controlsState.forcingSet.length > 0
                    ? controlsState.forcingSet.join(", ")
                    : "empty"}
                </p>
                <p>
                  <span className="font-medium">
                    Initial profile mode:
                  </span>{" "}
                  {controlsState.initialMode}
                </p>
                {controlsState.initialMode === "custom" && (
                  <p>
                    <span className="font-medium">
                      Custom active nodes:
                    </span>{" "}
                    {controlsState.customActiveNodes.length > 0
                      ? controlsState.customActiveNodes.join(", ")
                      : "none"}
                  </p>
                )}
                <p>
                  <span className="font-medium">
                    Target profile for forcing set:
                  </span>{" "}
                  {controlsState.targetProfile}
                </p>
                <p>
                  <span className="font-medium">
                    Max cascade steps:
                  </span>{" "}
                  {controlsState.maxSteps}
                </p>
                <p>
                  <span className="font-medium">
                    Compute PSNE:
                  </span>{" "}
                  {controlsState.showPsne ? "yes" : "no"}
                </p>
                <p>
                  <span className="font-medium">
                    Show interactive network:
                  </span>{" "}
                  {controlsState.showPyvis ? "yes" : "no"}
                </p>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

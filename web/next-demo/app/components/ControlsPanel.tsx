// web/next-demo/app/components/ControlsPanel.tsx
"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

export type NodeId = string;

export interface ExampleOption {
  key: string;
  name: string;
  description?: string;
}

export type InitialProfileMode = "default" | "all_inactive" | "custom";
export type TargetProfileMode = "all_inactive" | "all_active";

export interface ControlsState {
  selectedExampleKey: string;
  forcingSet: NodeId[];
  initialMode: InitialProfileMode;
  customActiveNodes: NodeId[];
  targetProfile: TargetProfileMode;
  maxSteps: number;
  showPsne: boolean;
  showPyvis: boolean;
}

export interface ControlsPanelProps {
  /**
   * Example games that the user can choose from.
   * These should correspond to presets in the backend or Python layer.
   */
  examples: ExampleOption[];

  /**
   * Node identifiers for the currently selected example.
   * The panel uses this to render checkboxes for forcing sets
   * and custom initial active nodes.
   */
  nodes: NodeId[];

  /**
   * Defaults for the initial state of the controls.
   * These let the parent align the panel with whatever example
   * instance the backend has created.
   */
  defaultSelectedExampleKey?: string;
  defaultForcingSet?: NodeId[];
  defaultInitialMode?: InitialProfileMode;
  defaultCustomActiveNodes?: NodeId[];
  defaultTargetProfile?: TargetProfileMode;
  defaultMaxSteps?: number;
  defaultShowPsne?: boolean;
  defaultShowPyvis?: boolean;

  /**
   * Callback that fires whenever the user clicks the "Run cascade" button.
   * The parent can trigger a simulation and re-render the graph.
   */
  onRunCascade: (state: ControlsState) => void;

  /**
   * Optional callback that fires whenever any control changes.
   * This is useful if the parent wants to keep the current settings
   * in a higher level state or synchronize with URL parameters.
   */
  onChange?: (state: ControlsState) => void;
}

function uniqueSortedNodes(nodes: NodeId[]): NodeId[] {
  return Array.from(new Set(nodes)).sort();
}

interface CheckboxListProps {
  label: string;
  nodes: NodeId[];
  selected: NodeId[];
  onChange: (nextSelected: NodeId[]) => void;
}

const CheckboxList: React.FC<CheckboxListProps> = ({
  label,
  nodes,
  selected,
  onChange,
}) => {
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const toggleNode = useCallback(
    (node: NodeId) => {
      const next = new Set(selectedSet);
      if (next.has(node)) {
        next.delete(node);
      } else {
        next.add(node);
      }
      onChange(Array.from(next).sort());
    },
    [selectedSet, onChange]
  );

  if (nodes.length === 0) {
    return (
      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium text-gray-800">{label}</span>
        <span className="text-xs text-gray-500">No nodes available</span>
      </div>
    );
  }

  return (
    <fieldset className="flex flex-col gap-1 border border-gray-200 rounded-md p-2">
      <legend className="text-sm font-medium text-gray-800 px-1">
        {label}
      </legend>
      <div className="flex flex-wrap gap-2">
        {nodes.map((node) => (
          <label
            key={node}
            className="inline-flex items-center gap-1 text-xs text-gray-800"
          >
            <input
              type="checkbox"
              className="h-3 w-3"
              checked={selectedSet.has(node)}
              onChange={() => toggleNode(node)}
            />
            <span>{node}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
};

export const ControlsPanel: React.FC<ControlsPanelProps> = ({
  examples,
  nodes,
  defaultSelectedExampleKey,
  defaultForcingSet,
  defaultInitialMode,
  defaultCustomActiveNodes,
  defaultTargetProfile,
  defaultMaxSteps,
  defaultShowPsne,
  defaultShowPyvis,
  onRunCascade,
  onChange,
}) => {
  const normalizedNodes = useMemo(
    () => uniqueSortedNodes(nodes),
    [nodes]
  );

  const initialExampleKey =
    defaultSelectedExampleKey ??
    (examples.length > 0 ? examples[0].key : "");

  const [selectedExampleKey, setSelectedExampleKey] =
    useState<string>(initialExampleKey);
  const [forcingSet, setForcingSet] = useState<NodeId[]>(
    defaultForcingSet ? [...defaultForcingSet] : []
  );
  const [initialMode, setInitialMode] = useState<InitialProfileMode>(
    defaultInitialMode ?? "default"
  );
  const [customActiveNodes, setCustomActiveNodes] = useState<NodeId[]>(
    defaultCustomActiveNodes ? [...defaultCustomActiveNodes] : []
  );
  const [targetProfile, setTargetProfile] = useState<TargetProfileMode>(
    defaultTargetProfile ?? "all_active"
  );
  const [maxSteps, setMaxSteps] = useState<number>(
    defaultMaxSteps ?? 10
  );
  const [showPsne, setShowPsne] = useState<boolean>(
    defaultShowPsne ?? (normalizedNodes.length <= 10)
  );
  const [showPyvis, setShowPyvis] = useState<boolean>(
    defaultShowPyvis ?? false
  );

  const currentState: ControlsState = useMemo(
    () => ({
      selectedExampleKey,
      forcingSet,
      initialMode,
      customActiveNodes,
      targetProfile,
      maxSteps,
      showPsne,
      showPyvis,
    }),
    [
      selectedExampleKey,
      forcingSet,
      initialMode,
      customActiveNodes,
      targetProfile,
      maxSteps,
      showPsne,
      showPyvis,
    ]
  );

  useEffect(() => {
    if (onChange) {
      onChange(currentState);
    }
  }, [currentState, onChange]);

  const handleExampleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const nextKey = e.target.value;
      setSelectedExampleKey(nextKey);
    },
    []
  );

  const handleMaxStepsChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = parseInt(e.target.value, 10);
      if (!Number.isNaN(value) && value > 0) {
        setMaxSteps(value);
      }
    },
    []
  );

  const handleRunClick = useCallback(() => {
    onRunCascade(currentState);
  }, [onRunCascade, currentState]);

  const selectedExample = useMemo(
    () => examples.find((ex) => ex.key === selectedExampleKey) || null,
    [examples, selectedExampleKey]
  );

  return (
    <aside className="flex flex-col gap-4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <header className="flex flex-col gap-1">
        <h2 className="text-base font-semibold text-gray-900">
          Game configuration
        </h2>
        <p className="text-xs text-gray-600">
          Choose an example game, forcing set, and initial conditions,
          then run a cascade.
        </p>
      </header>

      <section className="flex flex-col gap-2">
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium text-gray-800">Example game</span>
          <select
            value={selectedExampleKey}
            onChange={handleExampleChange}
            className="rounded-md border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            {examples.map((ex) => (
              <option key={ex.key} value={ex.key}>
                {ex.name}
              </option>
            ))}
          </select>
        </label>
        {selectedExample && selectedExample.description && (
          <p className="text-xs text-gray-600">
            {selectedExample.description}
          </p>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold text-gray-900">
          Dynamics configuration
        </h3>

        <CheckboxList
          label="Forcing set (nodes fixed exogenously)"
          nodes={normalizedNodes}
          selected={forcingSet}
          onChange={setForcingSet}
        />

        <fieldset className="flex flex-col gap-1 border border-gray-200 rounded-md p-2">
          <legend className="text-sm font-medium text-gray-800 px-1">
            Initial profile
          </legend>

          <div className="flex flex-col gap-1 text-xs text-gray-800">
            <label className="inline-flex items-center gap-1">
              <input
                type="radio"
                name="initial-profile-mode"
                value="default"
                checked={initialMode === "default"}
                onChange={() => setInitialMode("default")}
              />
              <span>Use example default</span>
            </label>

            <label className="inline-flex items-center gap-1">
              <input
                type="radio"
                name="initial-profile-mode"
                value="all_inactive"
                checked={initialMode === "all_inactive"}
                onChange={() => setInitialMode("all_inactive")}
              />
              <span>All inactive</span>
            </label>

            <label className="inline-flex items-center gap-1">
              <input
                type="radio"
                name="initial-profile-mode"
                value="custom"
                checked={initialMode === "custom"}
                onChange={() => setInitialMode("custom")}
              />
              <span>Custom active nodes</span>
            </label>
          </div>

          {initialMode === "custom" && (
            <div className="mt-2">
              <CheckboxList
                label="Nodes initially active"
                nodes={normalizedNodes}
                selected={customActiveNodes}
                onChange={setCustomActiveNodes}
              />
            </div>
          )}
        </fieldset>

        <fieldset className="flex flex-col gap-1 border border-gray-200 rounded-md p-2">
          <legend className="text-sm font-medium text-gray-800 px-1">
            Target profile for forcing set
          </legend>
          <div className="flex flex-col gap-1 text-xs text-gray-800">
            <label className="inline-flex items-center gap-1">
              <input
                type="radio"
                name="target-profile-mode"
                value="all_inactive"
                checked={targetProfile === "all_inactive"}
                onChange={() => setTargetProfile("all_inactive")}
              />
              <span>All inactive</span>
            </label>
            <label className="inline-flex items-center gap-1">
              <input
                type="radio"
                name="target-profile-mode"
                value="all_active"
                checked={targetProfile === "all_active"}
                onChange={() => setTargetProfile("all_active")}
              />
              <span>All active</span>
            </label>
          </div>
        </fieldset>

        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium text-gray-800">Max cascade steps</span>
          <input
            type="number"
            min={1}
            max={200}
            value={maxSteps}
            onChange={handleMaxStepsChange}
            className="w-24 rounded-md border border-gray-300 px-2 py-1 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          <span className="text-xs text-gray-500">
            Upper limit on best response iterations.
          </span>
        </label>

        <div className="flex flex-col gap-1 text-xs text-gray-800">
          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showPsne}
              onChange={(e) => setShowPsne(e.target.checked)}
            />
            <span>Compute PSNE (for small graphs)</span>
          </label>

          <label className="inline-flex items-center gap-1">
            <input
              type="checkbox"
              checked={showPyvis}
              onChange={(e) => setShowPyvis(e.target.checked)}
            />
            <span>Show interactive network view</span>
          </label>
        </div>
      </section>

      <button
        type="button"
        onClick={handleRunClick}
        className="mt-2 inline-flex items-center justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
      >
        Run cascade
      </button>
    </aside>
  );
};

export default ControlsPanel;
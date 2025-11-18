"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

type EdaResults = {
  numeric_columns?: string[];
  categorical_columns?: string[];
  missing_values?: Record<string, number>;
  missing_percentage?: Record<string, number>;
  summary_stats?: Record<string, any>;
  visualizations?: {
    numeric?: Record<string, string>;
    categorical?: Record<string, string>;
  };
};

type Props = {
  edaResults: EdaResults | null;
  performEDA: () => Promise<void>;
  loading: boolean;
  onNext?: () => void;
};

export default function EdaPanel({ edaResults, performEDA, loading, onNext }: Props) {
  const totalMissing = edaResults?.missing_values
    ? Object.values(edaResults.missing_values).reduce((a, b) => a + Number(b || 0), 0)
    : 0;

  return (
    <Card className="p-6">
      {/* <h2 className="text-2xl font-bold mb-4">Exploratory Data Analysis</h2> */}
      {/* <p className="text-muted-foreground mb-6">
        Analyze your dataset with automatic visualizations and statistics
      </p> */}

      {!edaResults ? (
        <div className="text-center py-8">
          <Button onClick={performEDA} disabled={loading} size="lg">
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>Run EDA Analysis</>
            )}
          </Button>
        </div>
      ) : (
        <div className="space-y-6">
          {edaResults.numeric_columns && edaResults.numeric_columns.length > 0 && (
            <Card className="p-4 border border-border">
              <h3 className="font-semibold mb-3">Summary Statistics</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Column</th>
                      <th className="text-right p-2">Mean</th>
                      <th className="text-right p-2">Std</th>
                      <th className="text-right p-2">Min</th>
                      <th className="text-right p-2">Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {edaResults.numeric_columns.slice(0, 10).map((col) => (
                      <tr key={col} className="border-b">
                        <td className="p-2 font-medium">{col}</td>
                        <td className="text-right p-2">
                          {edaResults.summary_stats?.[col]?.mean?.toFixed?.(2) ?? "N/A"}
                        </td>
                        <td className="text-right p-2">
                          {edaResults.summary_stats?.[col]?.std?.toFixed?.(2) ?? "N/A"}
                        </td>
                        <td className="text-right p-2">
                          {edaResults.summary_stats?.[col]?.min?.toFixed?.(2) ?? "N/A"}
                        </td>
                        <td className="text-right p-2">
                          {edaResults.summary_stats?.[col]?.max?.toFixed?.(2) ?? "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Visualizations */}
          {edaResults.visualizations && (
            <>
              {edaResults.visualizations.numeric && (
                <section>
                  <h3 className="text-lg font-semibold mb-3">Numeric Visualizations</h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    {Object.entries(edaResults.visualizations.numeric).map(([name, base64]) => (
                      <div key={name} className="p-2">
                        <h4 className="text-sm text-center mb-2">{name}</h4>
                        <img src={`data:image/png;base64,${base64}`} alt={name} className="border rounded shadow" />
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {edaResults.visualizations.categorical && (
                <section>
                  <h3 className="text-lg font-semibold mb-3">Categorical Visualizations</h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    {Object.entries(edaResults.visualizations.categorical).map(([name, base64]) => (
                      <div key={name} className="p-2">
                        <h4 className="text-sm text-center mb-2">{name}</h4>
                        <img src={`data:image/png;base64,${base64}`} alt={name} className="border rounded shadow" />
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      )}


    </Card>
  );
}

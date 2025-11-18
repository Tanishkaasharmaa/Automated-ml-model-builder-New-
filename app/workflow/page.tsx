"use client"


import { useState, useEffect } from "react"
import { Header } from "@/components/header"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import EdaPanel from "@/components/eda/edapanel"



import {
  BarChart3,
  CheckCircle2,
  Droplets,
  Target,
  Sliders,
  Zap,
  TrendingUp,
  Sparkles,
  Loader2,
  Download,
  AlertCircle,
} from "lucide-react"

const steps = [
  { id: "eda", label: "EDA", icon: BarChart3 },
  { id: "validate", label: "Validate", icon: CheckCircle2 },
  { id: "clean", label: "Clean", icon: Droplets },
  { id: "task", label: "Task Type", icon: Target },
  { id: "features", label: "Features", icon: Sliders },
  { id: "train", label: "Train", icon: Zap },
  { id: "evaluate", label: "Evaluate", icon: TrendingUp },
  { id: "predict", label: "Predict", icon: Sparkles },
  { id: "test", label: "Test", icon: BarChart3 },
]

export default function WorkflowPage() {
  const [currentStep, setCurrentStep] = useState("eda")
  const currentStepIndex = steps.findIndex((s) => s.id === currentStep)
  const progress = ((currentStepIndex + 1) / steps.length) * 100

  const [dataset, setDataset] = useState<any[]>([])
  const [datasetName, setDatasetName] = useState("")
  const [sessionId, setSessionId] = useState("")
  const [loading, setLoading] = useState(false)

  // EDA results
  const [edaResults, setEdaResults] = useState<any>(null)

  // Validation results
  const [validationResults, setValidationResults] = useState<any>(null)

  // Cleaning options
  const [cleaningStrategy, setCleaningStrategy] = useState("mean")
  const [cleaningResults, setCleaningResults] = useState<any>(null)

  // Task detection
  const [targetColumn, setTargetColumn] = useState("")
  const [taskType, setTaskType] = useState("")
  const [taskResults, setTaskResults] = useState<any>(null)

  // Feature selection
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
  const [featureResults, setFeatureResults] = useState<any>(null)

  // Training
  const [modelType, setModelType] = useState("random_forest")
  const [trainingResults, setTrainingResults] = useState<any>(null)

  // Evaluation
  const [evaluationResults, setEvaluationResults] = useState<any>(null)

  // Prediction
  const [predictionInput, setPredictionInput] = useState<any>({})
  const [predictionResult, setPredictionResult] = useState<any>(null)

  // Testing
  const [testFile, setTestFile] = useState<File | null>(null);

  interface ClassificationResults {
    metrics: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1_score?: number;
    };
    confusion_matrix?: number[][];
    sample_predictions: any[];
  }

  interface RegressionResults {
    metrics: {
      mse?: number;
      rmse?: number;
      mae?: number;
      r2?: number;
    };
    sample_predictions: any[];
  }

  type TestResults = ClassificationResults | RegressionResults;

  const [testResults, setTestResults] = useState<TestResults | null>(null);



  useEffect(() => {
    const data = localStorage.getItem("ml_dataset")
    const name = localStorage.getItem("ml_dataset_name")
    if (data) {
      setDataset(JSON.parse(data))
      setDatasetName(name || "dataset.csv")
      setSessionId(Math.random().toString(36).substring(7))
    }
  }, [])

  const callMLAPI = async (action: string, params: any = {}) => {
    setLoading(true)
    try {
      const response = await fetch("/api/ml", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          data: dataset,
          sessionId,
          ...params,
        }),
      })
      const result = await response.json()
      if (result.error) {
        throw new Error(result.error)
      }
      return result
    } catch (error: any) {
      console.error("[v0] ML API Error:", error)
      alert(`Error: ${error.message}`)
      return null
    } finally {
      setLoading(false)
    }
  }

  const performEDA = async () => {
    const result = await callMLAPI("eda")
    if (result) setEdaResults(result)
  }

  const performValidation = async () => {
    const result = await callMLAPI("validate")
    if (result) setValidationResults(result)
  }

  const performCleaning = async () => {
    const result = await callMLAPI("clean", { strategy: cleaningStrategy })
    if (result) setCleaningResults(result)
  }

  // const performCleaning = async () => {
  //   const result = await callMLAPI("clean", { strategy: cleaningStrategy })
  //   if (result) {
  //     setCleaningResults(result)
  //     setDataset(result.cleaned_data) // update dataset for future API calls
  //   }
  // }


  const detectTask = async () => {
    if (!targetColumn) {
      alert("Please select a target column")
      return
    }
    const result = await callMLAPI("detect_task", { targetColumn })
    if (result) {
      setTaskResults(result)
      setTaskType(result.task_type)
    }
  }

  const selectFeatures = async () => {
    if (!targetColumn) {
      alert("Please select a target column first")
      return
    }
    const result = await callMLAPI("select_features", { targetColumn, nFeatures: 10 })
    if (result) {
      setFeatureResults(result)
      setSelectedFeatures(result.selected_features || [])
    }
  }

  const trainModel = async () => {
    const result = await callMLAPI("train", { modelType, taskType, targetColumn })
    if (result) setTrainingResults(result)
  }

  const evaluateModel = async () => {
    const result = await callMLAPI("evaluate", { taskType })
    if (result) setEvaluationResults(result)
  }

  const makePrediction = async () => {
    const result = await callMLAPI("predict", { inputData: predictionInput, taskType })
    if (result) setPredictionResult(result)
  }


    const runTestEvaluation = async () => {
    if (!testFile) {
      alert("Please upload a dataset first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", testFile);
    formData.append("session_id", sessionId);   // <-- REQUIRED

    const response = await fetch("http://localhost:8000/test-model", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setTestResults(data);
  };


  const columns = dataset.length > 0 ? Object.keys(dataset[0]) : []

  const downloadModel = async () => {
    try {

      const response = await fetch("http://localhost:8000/download_model");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "trained_model.pkl"; // File name
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Model download failed:", error);
      alert("Failed to download model. Make sure the model is trained.");
    }
  };

  return (
    <div className="min-h-screen">
      <Header />

      <div className="container py-8 max-w-7xl">
        {/* Progress Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold">ML Workflow Dashboard</h1>
              <p className="text-sm text-muted-foreground mt-1">
                Dataset: {datasetName} ({dataset.length} rows, {columns.length} columns)
              </p>
            </div>
            <div className="text-sm text-muted-foreground">
              Step {currentStepIndex + 1} of {steps.length}
            </div>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Step Navigation */}
        <div className="mb-8 overflow-x-auto">
          <div className="flex gap-2 min-w-max">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = step.id === currentStep
              const isCompleted = index < currentStepIndex

              return (
                <button
                  key={step.id}
                  onClick={() => setCurrentStep(step.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : isCompleted
                        ? "bg-accent text-accent-foreground"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="font-medium">{step.label}</span>
                </button>
              )
            })}
          </div>
        </div>

        <Tabs value={currentStep} onValueChange={setCurrentStep}>
          <TabsContent value="eda" className="mt-0">
          <h2 className="text-2xl font-bold mb-4">Exploratory Data Analysis</h2>
          {/* 1. Overview Cards — always at top */}
            {edaResults && (
              <div className="grid md:grid-cols-3 gap-4 mb-8">
                <Card className="p-4 border border-border shadow-sm">
                  <h3 className="text-sm text-muted-foreground font-bold">Rows</h3>
                  <p className="text-2xl font-bold">{edaResults.row_count}</p>
                </Card>

                <Card className="p-4 border border-border shadow-sm">
                  <h3 className="text-sm text-muted-foreground font-bold">Columns</h3>
                  <p className="text-2xl font-bold">{edaResults.column_count}</p>
                </Card>

                <Card className="p-4 border border-border shadow-sm">
                  <h3 className="text-sm text-muted-foreground font-bold ">Total Missing Values</h3>
                  <p className="text-2xl font-bold">{edaResults.total_missing_values}</p>
                </Card>
              </div>
            )}
            {edaResults && (
                  <div className="grid grid-cols-1 md:grid-cols-1 gap-6 mb-8">

                   {/* Dataset Overview */}
                    <Card className="p-6">
                      <h3 className="text-lg font-semibold mb-3">Dataset Overview</h3>

                      <div className="grid grid-cols-2 gap-y-2 text-sm">
                        <div className="font-bold text-muted-foreground">Numeric Columns:</div>
                        <div className="text-right font-bold">{edaResults.numeric_columns.length}</div>

                        <div className="font-bold text-muted-foreground">Categorical Columns:</div>
                        <div className="text-right font-bold">{edaResults.categorical_columns.length}</div>

                        <div className="font-bold text-muted-foreground">Total Missing Values:</div>
                        <div className="text-right font-bold">{edaResults.total_missing_values}</div>
                      </div>
                    </Card>
                  </div>
                )}

            {edaResults?.missing_values_pie_chart && edaResults?.missing_values_table && (
              <div className="flex flex-col md:flex-row gap-6 mb-8">

            {/* Missing Values Section (Side-by-Side) */}
                {edaResults?.missing_values_pie_chart && (
                  <div className="mb-8">
                    <h3 className="text-lg font-semibold mb-2">Missing Values Percentage</h3>
                    <img
                      src={`data:image/png;base64,${edaResults.missing_values_pie_chart}`}
                      alt="Missing values pie chart"
                      className="rounded-lg border w-full max-w-[600px] h-100"
                    />
                  </div>
                )}

                {/* Table */}
                <Card className="w-full md:w-2/3 p-6">
                  <h3 className="text-xl font-semibold mb-4">Missing Values Table</h3>

                  <div className="overflow-x-auto">
                    <table className="min-w-full border-collapse border border-gray-200 rounded-lg">
                      <thead>
                        <tr className="bg-gray-100 text-left">
                          <th className="border px-4 py-2">Column</th>
                          <th className="border px-4 py-2">Missing Count</th>
                          <th className="border px-4 py-2">Missing Percentage</th>
                        </tr>
                      </thead>

                      <tbody>
                        {edaResults.missing_values_table.map((row: any) => (
                          <tr key={row.column}>
                            <td className="border px-4 py-2 font-medium">{row.column}</td>
                            <td className="border px-4 py-2">{row.missing_count}</td>
                            <td className="border px-4 py-2">{row.missing_percentage}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

              </div>
            )}


            {/* 2. EDA Main Panel */}
            <EdaPanel
              edaResults={edaResults}
              performEDA={performEDA}
              loading={loading}
              onNext={() => setCurrentStep("validate")}
            />

            {edaResults?.distribution_shapes && (
              <Card className="p-6 mb-8">
                <h3 className="text-xl font-semibold mb-4">Distribution Shape (Skewness)</h3>

                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse border border-gray-200 rounded-lg">
                    <thead>
                      <tr className="bg-gray-100 text-left">
                        <th className="border px-4 py-2">Feature</th>
                        <th className="border px-4 py-2">Skewness</th>
                        <th className="border px-4 py-2">Shape</th>
                      </tr>
                    </thead>

                    <tbody>
                      {Object.entries(edaResults.distribution_shapes).map(([col, info]: any) => (
                        <tr key={col}>
                          <td className="border px-4 py-2 font-medium">{col}</td>
                          <td className="border px-4 py-2">{info.skewness.toFixed(3)}</td>
                          <td className="border px-4 py-2">{info.shape}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}


            {edaResults?.heatmap && (
              <div className="mb-8">
                <h3 className="text-lg font-semibold mb-2">Correlation Heatmap</h3>
                <img
                  src={`data:image/png;base64,${edaResults.heatmap}`}
                  alt="Correlation Heatmap"
                  className="rounded-lg border w-full max-w-[700px] h-auto"
                  style={{ width: "600px", maxWidth: "700px", height: "auto" }}
                />
              </div>
            )}

            {/* 4. NEXT BUTTON — always at end */}
            <div className="mt-6 flex justify-end">
              <Button onClick={() => setCurrentStep("validate")}>
                Next: Data Validation
              </Button>
            </div>

          </TabsContent>

          <TabsContent value="validate" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Data Validation</h2>
              <p className="text-muted-foreground mb-6">Detect errors, duplicates, and anomalies in your dataset</p>

              {!validationResults ? (
                <div className="text-center py-8">
                  <Button onClick={performValidation} disabled={loading} size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Validating...
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="mr-2 h-4 w-4" />
                        Run Validation
                      </>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Card className="p-4 border border-border">
                    <div className="flex items-start gap-3">
                      <AlertCircle
                        className={`h-5 w-5 mt-0.5 ${validationResults.duplicates > 0 ? "text-destructive" : "text-green-500"}`}
                      />
                      <div className="flex-1">
                        <h3 className="font-semibold mb-2">Duplicate Detection</h3>
                        <p className="text-sm text-muted-foreground">
                          Found <span className="font-medium">{validationResults.duplicates}</span> duplicate rows
                        </p>
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4 border border-border">
                    <div className="flex items-start gap-3">
                      <AlertCircle
                        className={`h-5 w-5 mt-0.5 ${validationResults.total_missing > 0 ? "text-yellow-500" : "text-green-500"}`}
                      />
                      <div className="flex-1">
                        <h3 className="font-semibold mb-2">Missing Values</h3>
                        <p className="text-sm text-muted-foreground mb-2">
                          Total missing values: <span className="font-medium">{validationResults.total_missing}</span>
                        </p>
                        {validationResults.columns_with_missing?.length > 0 && (
                          <p className="text-sm text-muted-foreground">
                            Columns affected: {validationResults.columns_with_missing.join(", ")}
                          </p>
                        )}
                      </div>
                    </div>
                  </Card>

                  <Card className="p-4 border border-border">
                    <h3 className="font-semibold mb-3">Outlier Detection (IQR Method)</h3>
                    <div className="space-y-2 text-sm max-h-40 overflow-y-auto">
                      {Object.entries(validationResults.outliers || {}).map(([col, count]: any) => (
                        <div key={col} className="flex justify-between">
                          <span className="text-muted-foreground">{col}:</span>
                          <span className={count > 0 ? "text-yellow-600 font-medium" : "text-green-600"}>
                            {count} outliers
                          </span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              )}

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("eda")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("clean")}>Next: Clean Data</Button>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="clean" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Clean Data</h2>
              <p className="text-muted-foreground mb-6">Handle missing values and prepare data for modeling</p>

              <div className="space-y-6">
                <div className="space-y-3">
                  <Label>Missing Value Strategy</Label>
                  <Select value={cleaningStrategy} onValueChange={setCleaningStrategy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mean">Fill with Mean</SelectItem>
                      <SelectItem value="median">Fill with Median</SelectItem>
                      <SelectItem value="drop">Drop Rows with Missing Values</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {!cleaningResults ? (
                  <Button onClick={performCleaning} disabled={loading}>
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Cleaning...
                      </>
                    ) : (
                      <>
                        <Droplets className="mr-2 h-4 w-4" />
                        Apply Cleaning
                      </>
                    )}
                  </Button>
                ) : (
                  <Card className="p-4 border border-green-500/50 bg-green-500/5">
                    <h3 className="font-semibold mb-3 text-green-700 dark:text-green-400">Cleaning Complete</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Original rows:</span>
                        <span className="font-medium">{cleaningResults.original_shape?.[0]}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Cleaned rows:</span>
                        <span className="font-medium">{cleaningResults.new_shape?.[0]}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Rows removed:</span>
                        <span className="font-medium text-destructive">{cleaningResults.rows_removed}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Strategy:</span>
                        <span className="font-medium capitalize">{cleaningResults.strategy_used}</span>
                      </div>
                    </div>
                  </Card>
                )}
              </div>

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("validate")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("task")}>Next: Choose Task</Button>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="task" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Choose Task Type</h2>
              <p className="text-muted-foreground mb-6">Select your target column to auto-detect the task type</p>

              <div className="space-y-6">
                <div className="space-y-3">
                  <Label>Target Column (What do you want to predict?)</Label>
                  <Select value={targetColumn} onValueChange={setTargetColumn}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select target column" />
                    </SelectTrigger>
                    <SelectContent>
                      {columns.map((col) => (
                        <SelectItem key={col} value={col}>
                          {col}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {!taskResults ? (
                  <Button onClick={detectTask} disabled={loading || !targetColumn}>
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Detecting...
                      </>
                    ) : (
                      <>
                        <Target className="mr-2 h-4 w-4" />
                        Detect Task Type
                      </>
                    )}
                  </Button>
                ) : (
                  <Card className="p-4 border border-primary/50 bg-primary/5">
                    <h3 className="font-semibold mb-3">Task Detected</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Task Type:</span>
                        <span className="font-medium capitalize">{taskResults.task_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Target Column:</span>
                        <span className="font-medium">{taskResults.target_column}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Unique Values:</span>
                        <span className="font-medium">{taskResults.unique_values}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Is Numeric:</span>
                        <span className="font-medium">{taskResults.is_numeric ? "Yes" : "No"}</span>
                      </div>
                    </div>
                  </Card>
                )}
              </div>

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("clean")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("features")} disabled={!taskResults}>
                  Next: Feature Selection
                </Button>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="features" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Feature Selection</h2>
              <p className="text-muted-foreground mb-6">Select the most important features for your model</p>

              {!featureResults ? (
                <div className="text-center py-8">
                  <Button onClick={selectFeatures} disabled={loading || !targetColumn} size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing Features...
                      </>
                    ) : (
                      <>
                        <Sliders className="mr-2 h-4 w-4" />
                        Run Feature Selection
                      </>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Card className="p-4 border border-border">
                    <h3 className="font-semibold mb-3">Feature Importance Scores</h3>
                    <div className="space-y-2 max-h-80 overflow-y-auto">
                      {featureResults.feature_scores?.slice(0, 15).map((item: any, idx: number) => (
                        <div key={item.feature} className="flex items-center gap-3">
                          <span className="text-xs text-muted-foreground w-6">{idx + 1}</span>
                          <div className="flex-1">
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">{item.feature}</span>
                              <span className="text-sm text-muted-foreground">{item.score?.toFixed(2)}</span>
                            </div>
                            <Progress
                              value={(item.score / featureResults.feature_scores[0].score) * 100}
                              className="h-2"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>

                  <Card className="p-4 border border-green-500/50 bg-green-500/5">
                    <h3 className="font-semibold mb-2 text-green-700 dark:text-green-400">Selected Features</h3>
                    <p className="text-sm text-muted-foreground mb-3">
                      Top {featureResults.n_features_selected} features selected for training
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {selectedFeatures.map((feature) => (
                        <span key={feature} className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </Card>
                </div>
              )}

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("task")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("train")} disabled={!featureResults}>
                  Next: Train Model
                </Button>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="train" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Train Model</h2>
              <p className="text-muted-foreground mb-6">Select and train your machine learning model</p>

              <div className="space-y-6">
                <div className="space-y-3">
                  <Label>Model Type</Label>
                  <Select value={modelType} onValueChange={setModelType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {taskType === "classification" ? (
                        <>
                          <SelectItem value="random_forest">Random Forest</SelectItem>
                          <SelectItem value="logistic_regression">Logistic Regression</SelectItem>
                          <SelectItem value="svm">Support Vector Machine (SVM)</SelectItem>
                          <SelectItem value="decision_tree">Decision Tree</SelectItem>
                          <SelectItem value="knn">K-Nearest Neighbors (KNN)</SelectItem>
                        </>
                      ) : (
                        <>
                          <SelectItem value="random_forest">Random Forest Regressor</SelectItem>
                          <SelectItem value="linear_regression">Linear Regression</SelectItem>
                          <SelectItem value="svm">Support Vector Regressor (SVR)</SelectItem>
                          <SelectItem value="decision_tree">Decision Tree Regressor</SelectItem>
                          <SelectItem value="knn">KNN Regressor</SelectItem>
                        </>
                      )}

                    </SelectContent>
                  </Select>
                </div>

                {!trainingResults ? (
                  <Button onClick={trainModel} disabled={loading || !taskType} size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Training Model...
                      </>
                    ) : (
                      <>
                        <Zap className="mr-2 h-4 w-4" />
                        Start Training
                      </>
                    )}
                  </Button>
                ) : (
                  <Card className="p-4 border border-green-500/50 bg-green-500/5">
                    <h3 className="font-semibold mb-3 text-green-700 dark:text-green-400">Training Complete!</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model Type:</span>
                        <span className="font-medium capitalize">{trainingResults.model_type?.replace("_", " ")}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Task Type:</span>
                        <span className="font-medium capitalize">{trainingResults.task_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Status:</span>
                        <span className="font-medium text-green-600">Ready for Evaluation</span>
                      </div>
                    </div>
                  </Card>
                )}
              </div>

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("features")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("evaluate")} disabled={!trainingResults}>
                  Next: Evaluate Model
                </Button>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="evaluate" className="mt-0">
            < Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Model Evaluation</h2>
              <p className="text-muted-foreground mb-6">Review your model's performance metrics</p>

              {!evaluationResults ? (
                <div className="text-center py-8">
                  <Button onClick={evaluateModel} disabled={loading || !trainingResults} size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Evaluating...
                      </>
                    ) : (
                      <>
                        <TrendingUp className="mr-2 h-4 w-4" />
                        Evaluate Model
                      </>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-6">
                  {("test_accuracy" in evaluationResults) ? (
                    // Classification Metrics
                    <>
                      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <Card className="p-4 border border-border">
                          <div className="text-sm text-muted-foreground mb-1">Accuracy</div>
                          <div className="text-2xl font-bold">{(evaluationResults.test_accuracy * 100).toFixed(2)}%</div>
                        </Card>
                        <Card className="p-4 border border-border">
                          <div className="text-sm text-muted-foreground mb-1">Precision</div>
                          <div className="text-2xl font-bold">{(evaluationResults.precision * 100).toFixed(2)}%</div>
                        </Card>
                        <Card className="p-4 border border-border">
                          <div className="text-sm text-muted-foreground mb-1">Recall</div>
                          <div className="text-2xl font-bold">{(evaluationResults.recall * 100).toFixed(2)}%</div>
                        </Card>
                        <Card className="p-4 border border-border">
                          <div className="text-sm text-muted-foreground mb-1">F1 Score</div>
                          <div className="text-2xl font-bold">{(evaluationResults.f1_score * 100).toFixed(2)}%</div>
                        </Card>
                      </div>
                    </>
                  ) : (
                    // Regression Metrics
                    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <Card className="p-4 border border-border">
                        <div className="text-sm text-muted-foreground mb-1">RMSE</div>
                        <div className="text-2xl font-bold">{evaluationResults.rmse.toFixed(4)}</div>
                      </Card>
                      <Card className="p-4 border border-border">
                        <div className="text-sm text-muted-foreground mb-1">MSE</div>
                        <div className="text-2xl font-bold">{evaluationResults.mse.toFixed(4)}</div>
                      </Card>
                      <Card className="p-4 border border-border">
                        <div className="text-sm text-muted-foreground mb-1">MAE</div>
                        <div className="text-2xl font-bold">{evaluationResults.mae.toFixed(4)}</div>
                      </Card>
                      <Card className="p-4 border border-border">
                        <div className="text-sm text-muted-foreground mb-1">R² Score</div>
                        <div className="text-2xl font-bold">{evaluationResults.r2.toFixed(4)}</div>
                      </Card>
                    </div>
                  )}

                  {/* Sample Predictions */}
                  {evaluationResults.sample_predictions && evaluationResults.sample_predictions.length > 0 && (
                    <Card className="p-4 border border-border mt-4">
                      <h3 className="font-semibold mb-2">Sample Predictions</h3>
                      <div className="flex flex-wrap gap-2">
                        {evaluationResults.sample_predictions.map((val: string | number, idx: number) => (
                          <div key={idx} className="px-3 py-1 bg-pink-100 text-pink-700 rounded">
                            {val}
                          </div>
                        ))}
                      </div>
                    </Card>
                  )}
                </div>
              )}
              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("train")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("predict")} disabled={!evaluationResults}>
                  Next: Make Predictions
                </Button>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="predict" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Make Predictions</h2>
              <p className="text-muted-foreground mb-6">Use your trained model to make predictions on new data</p>

              <div className="space-y-6">
                <Card className="p-4 border border-border">
                  <h3 className="font-semibold mb-4">Input Features</h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    {selectedFeatures.map((feature) => (
                      <div key={feature} className="space-y-2">
                        <Label htmlFor={feature}>{feature}</Label>
                        <Input
                          id={feature}
                          placeholder={`Enter ${feature}`}
                          value={predictionInput[feature] || ""}
                          onChange={(e) => setPredictionInput({ ...predictionInput, [feature]: e.target.value })}
                        />
                      </div>
                    ))}
                  </div>
                  <Button onClick={makePrediction} disabled={loading} className="mt-4">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Predicting...
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-2 h-4 w-4" />
                        Generate Prediction
                      </>
                    )}
                  </Button>
                </Card>

                {predictionResult && (
                  <Card className="p-4 border border-primary/50 bg-primary/5">
                    <h3 className="font-semibold mb-3">Prediction Result</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Predicted Value:</span>
                        <span className="text-2xl font-bold text-primary">{predictionResult.prediction}</span>
                      </div>
                      {predictionResult.confidence && (
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Confidence:</span>
                          <span className="font-medium">{(predictionResult.confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  </Card>
                )}
              </div> 

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("evaluate")}>
                  Back
                </Button>
                <Button onClick={() => setCurrentStep("test")}>
                  Next: Test model
                </Button>
              </div>
            </Card>
          </TabsContent>
          <TabsContent value="test">
              <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Upload Dataset for Testing</h2>
              <p className="text-sm text-muted-foreground mb-4">
                Upload a dataset to evaluate the exported model.
              </p>

              <div className="flex items-center space-x-4 mb-4">
                {/* Hidden file input */}
                <input
                  id="test-file-upload"
                  type="file"
                  accept=".csv"
                  onChange={(e) => setTestFile(e.target.files?.[0] ?? null)}
                  className="hidden"
                />

                {/* Custom upload button */}
                <label
                  htmlFor="test-file-upload"
                  className="cursor-pointer bg-pink-400 text-white px-4 py-2 rounded hover:bg-pink-500"
                >
                  {testFile ? "Change File" : "Upload CSV"}
                </label>

                {/* Display selected file name */}
                {testFile && <span className="text-gray-700 font-medium">{testFile.name}</span>}
              </div>

              <Button onClick={runTestEvaluation} disabled={!testFile}>
                Run Evaluation
              </Button>

                {testResults && (
  <div className="space-y-6">
    {/* Metrics */}
    {testResults.metrics && (
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(testResults.metrics).map(([key, value]) =>
          value !== undefined ? (
            <Card key={key} className="p-4 border border-border">
              <div className="text-sm text-muted-foreground mb-1">
                {key.replace(/_/g, " ").toUpperCase()}
              </div>
              <div className="text-2xl font-bold">
                {typeof value === "number"
                  ? ["accuracy", "precision", "recall", "f1_score"].includes(key)
                    ? (value * 100).toFixed(2) + "%"
                    : value.toFixed(2)
                  : String(value)}
              </div>
            </Card>
          ) : null
        )}
      </div>
    )}

    {/* Confusion Matrix (only for classification) */}
    {"confusion_matrix" in testResults && testResults.confusion_matrix && (
      <Card className="p-4 border border-border">
        <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full table-auto border-collapse border border-gray-200">
            <thead>
              <tr>
                <th className="border px-4 py-2 bg-gray-100">Actual \ Predicted</th>
                {testResults.confusion_matrix[0].map((_, idx) => (
                  <th key={idx} className="border px-4 py-2 bg-gray-100">
                    {idx}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {testResults.confusion_matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td className="border px-4 py-2 font-medium">{rowIdx}</td>
                  {row.map((val, colIdx) => (
                    <td
                      key={colIdx}
                      className={`border px-4 py-2 text-center ${
                        rowIdx === colIdx ? "bg-green-100" : ""
                      }`}
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )}

   


                    {/* Sample Predictions */}
                    {testResults.sample_predictions && testResults.sample_predictions.length > 0 && (
                      <Card className="p-4 border border-border">
                        <h3 className="font-semibold mb-2">Sample Predictions</h3>
                        <pre className="text-sm text-gray-700 overflow-x-auto">
                          {JSON.stringify(testResults.sample_predictions, null, 2)}
                        </pre>
                      </Card>
                    )}

                  </div>
                )}
                {/* Export Model */}
                <Card className="p-6 border shadow-sm">
                  <h3 className="text-lg font-semibold mb-3">Export Model</h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Download your trained model (.pkl) for future predictions
                  </p>
                  <Button variant="outline" onClick={downloadModel}>
                    <Download className="mr-2 h-4 w-4" />
                    Download Model (.pkl)
                  </Button>
                </Card>
              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("predict")}>
                  Back
                </Button>
                <Button onClick={() => alert("Workflow complete! You can now download your model .")}>
                  Complete Workflow
                </Button>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

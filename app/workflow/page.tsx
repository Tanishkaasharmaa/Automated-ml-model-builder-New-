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

  const downloadModel = () => {
    const modelData = {
      sessionId,
      taskType,
      modelType,
      targetColumn,
      selectedFeatures,
      trainingResults,
      evaluationResults,
    }
    const blob = new Blob([JSON.stringify(modelData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `ml_model_${sessionId}.json`
    a.click()
  }

  const columns = dataset.length > 0 ? Object.keys(dataset[0]) : []

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

        {/* Content Area */}
        <Tabs value={currentStep} onValueChange={setCurrentStep}>
          <TabsContent value="eda" className="mt-0">
            <Card className="p-6">
              <h2 className="text-2xl font-bold mb-4">Exploratory Data Analysis</h2>
              <p className="text-muted-foreground mb-6">
                Analyze your dataset with automatic visualizations and statistics
              </p>

              {!edaResults ? (
                <div className="text-center py-8">
                  <Button onClick={performEDA} disabled={loading} size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <BarChart3 className="mr-2 h-4 w-4" />
                        Run EDA Analysis
                      </>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <Card className="p-4 border border-border">
                      <h3 className="font-semibold mb-3">Dataset Overview</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Numeric Columns:</span>
                          <span className="font-medium">{edaResults.numeric_columns?.length || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Categorical Columns:</span>
                          <span className="font-medium">{edaResults.categorical_columns?.length || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Missing Values:</span>
                          <span className="font-medium">
                            {edaResults?.missing_values
                              ? Object.values(edaResults.missing_values).reduce(
                                (a: number, b: any) => a + (typeof b === "number" ? b : 0),
                                0
                              )
                            : 0}
                          </span>
                        </div>
                      </div>
                    </Card>

                    <Card className="p-4 border border-border">
                      <h3 className="font-semibold mb-3">Missing Values by Column</h3>
                      <div className="space-y-2 text-sm max-h-40 overflow-y-auto">
                        {Object.entries(edaResults.missing_values || {}).map(([col, count]: any) => (
                          <div key={col} className="flex justify-between">
                            <span className="text-muted-foreground truncate">{col}:</span>
                            <span className={count > 0 ? "text-destructive font-medium" : "text-muted-foreground"}>
                              {count} ({edaResults.missing_percentage?.[col]?.toFixed(1)}%)
                            </span>
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>

                  {edaResults.numeric_columns?.length > 0 && (
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
                            {edaResults.numeric_columns.slice(0, 5).map((col: string) => (
                              <tr key={col} className="border-b">
                                <td className="p-2 font-medium">{col}</td>
                                <td className="text-right p-2">
                                  {edaResults.summary_stats?.[col]?.mean?.toFixed(2) || "N/A"}
                                </td>
                                <td className="text-right p-2">
                                  {edaResults.summary_stats?.[col]?.std?.toFixed(2) || "N/A"}
                                </td>
                                <td className="text-right p-2">
                                  {edaResults.summary_stats?.[col]?.min?.toFixed(2) || "N/A"}
                                </td>
                                <td className="text-right p-2">
                                  {edaResults.summary_stats?.[col]?.max?.toFixed(2) || "N/A"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </Card>
                  )}
                </div>
              )}

              <div className="mt-6 flex justify-end">
                <Button onClick={() => setCurrentStep("validate")}>Next: Data Validation</Button>
              </div>
            </Card>
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
                          <SelectItem value="random_forest">Random Forest Classifier</SelectItem>
                          <SelectItem value="logistic_regression">Logistic Regression</SelectItem>
                          <SelectItem value="decision_tree">Decision Tree Classifier</SelectItem>
                        </>
                      ) : (
                        <>
                          <SelectItem value="random_forest">Random Forest Regressor</SelectItem>
                          <SelectItem value="linear_regression">Linear Regression</SelectItem>
                          <SelectItem value="decision_tree">Decision Tree Regressor</SelectItem>
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
            <Card className="p-6">
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

                  <Card className="p-4 border border-border">
                    <h3 className="font-semibold mb-3">Performance Summary</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between mb-1 text-sm">
                          <span>Overall Accuracy</span>
                          <span className="font-medium">{(evaluationResults.test_accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={evaluationResults.test_accuracy * 100} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between mb-1 text-sm">
                          <span>Precision</span>
                          <span className="font-medium">{(evaluationResults.precision * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={evaluationResults.precision * 100} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between mb-1 text-sm">
                          <span>Recall</span>
                          <span className="font-medium">{(evaluationResults.recall * 100).toFixed(1)}%</span>
                        </div>
                        <Progress value={evaluationResults.recall * 100} className="h-2" />
                      </div>
                    </div>
                  </Card>
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

                <Card className="p-4 border border-border">
                  <h3 className="font-semibold mb-3">Export Model</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Download your trained model configuration and results
                  </p>
                  <Button variant="outline" onClick={downloadModel}>
                    <Download className="mr-2 h-4 w-4" />
                    Download Model (.json)
                  </Button>
                </Card>
              </div>

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="outline" onClick={() => setCurrentStep("evaluate")}>
                  Back
                </Button>
                <Button
                  onClick={() => alert("Workflow complete! You can now download your model or make more predictions.")}
                >
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

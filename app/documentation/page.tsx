import { Header } from "@/components/header"
import { Card } from "@/components/ui/card"
import { Upload, BarChart3, CheckCircle2, Droplets, Target, Sliders, Zap, TrendingUp, Sparkles } from "lucide-react"

export default function DocumentationPage() {
  return (
    <div className="min-h-screen">
      <Header />

      <div className="container py-12 max-w-5xl">
        <h1 className="text-4xl font-bold mb-6">Documentation</h1>
        <p className="text-lg text-muted-foreground mb-8">A comprehensive guide to using AutoML Builder</p>

        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 shrink-0">
                <Upload className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">1. Upload Dataset</h2>
                <p className="text-muted-foreground">
                  Start by uploading your dataset in CSV, JSON, or Excel format. You can drag and drop your file or
                  click to browse. A preview of the first 5 rows will be displayed to confirm your data loaded
                  correctly. You can also use our sample dataset to explore the platform.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 shrink-0">
                <BarChart3 className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">2. Exploratory Data Analysis (EDA)</h2>
                <p className="text-muted-foreground">
                  The platform automatically generates summary statistics, visualizations, and identifies missing values
                  in your dataset. All charts and metrics are computed dynamically from your data.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 shrink-0">
                <CheckCircle2 className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">3. Data Validation</h2>
                <p className="text-muted-foreground">
                  Automatically detect errors, duplicates, and anomalies in your dataset. The system will flag potential
                  issues and suggest corrections.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 shrink-0">
                <Droplets className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">4. Clean Data</h2>
                <p className="text-muted-foreground">
                  Choose how to handle missing values (fill with mean/median or drop rows) and encode categorical
                  features using one-hot or label encoding.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary/10 shrink-0">
                <Target className="h-5 w-5 text-secondary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">5. Choose Task Type</h2>
                <p className="text-muted-foreground">
                  Select whether you want to perform Classification (predict categories), Regression (predict continuous
                  values), or Clustering (group similar data points).
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary/10 shrink-0">
                <Sliders className="h-5 w-5 text-secondary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">6. Feature Selection</h2>
                <p className="text-muted-foreground">
                  Review feature importance scores and select which features to include in your model. The platform
                  helps identify the most relevant variables for prediction.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary/10 shrink-0">
                <Zap className="h-5 w-5 text-secondary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">7. Train Model</h2>
                <p className="text-muted-foreground">
                  Watch your model train in real-time with progress indicators and live metrics. The training process is
                  fully automated and optimized.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary/10 shrink-0">
                <TrendingUp className="h-5 w-5 text-secondary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">8. Model Evaluation</h2>
                <p className="text-muted-foreground">
                  Review comprehensive metrics including Accuracy, Precision, Recall, F1 Score, RMSE, and more.
                  Visualize results with confusion matrices and performance charts.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary/10 shrink-0">
                <Sparkles className="h-5 w-5 text-secondary" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">9. Make Predictions</h2>
                <p className="text-muted-foreground">
                  Input new data to get real predictions from your trained model. Download your model as a pickle (.pkl)
                  file for use in other applications. Export a complete pipeline report as PDF or CSV.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

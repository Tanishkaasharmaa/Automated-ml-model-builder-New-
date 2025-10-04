import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Header } from "@/components/header"
import { Upload, BarChart3, Settings, Sparkles, Target, Download } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <Header />

      {/* Hero Section */}
      <section className="container py-24 md:py-32">
        <div className="mx-auto max-w-4xl text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-accent px-4 py-1.5 text-sm font-medium text-accent-foreground">
            <Sparkles className="h-4 w-4" />
            Version 1.0 — Minimal Viable Product
          </div>

          <h1 className="mb-6 text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-balance">
            Automated ML Model Builder
          </h1>

          <p className="mb-8 text-xl md:text-2xl text-muted-foreground text-balance">
            Upload your dataset → Explore → Build a model → Predict
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" className="text-base">
              <Link href="/upload">Get Started</Link>
            </Button>
            <Button asChild size="lg" variant="outline" className="text-base bg-transparent">
              <Link href="/documentation">View Documentation</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Workflow Steps */}
      <section className="container py-16 md:py-24 bg-muted/30">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-12 text-3xl md:text-4xl font-bold text-center">Simple, Guided Workflow</h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Upload className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Upload Dataset</h3>
              <p className="text-muted-foreground">
                Support for CSV, JSON, and Excel formats. Preview your data instantly.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Explore & Validate</h3>
              <p className="text-muted-foreground">Automatic EDA with charts, statistics, and anomaly detection.</p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Settings className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Clean & Prepare</h3>
              <p className="text-muted-foreground">
                Handle missing values, encode features, and select important variables.
              </p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-secondary/10">
                <Target className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Choose Task</h3>
              <p className="text-muted-foreground">Classification, Regression, or Clustering — we guide you through.</p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-secondary/10">
                <Sparkles className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Train & Evaluate</h3>
              <p className="text-muted-foreground">Dynamic training with real-time metrics and model evaluation.</p>
            </Card>

            <Card className="p-6 hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-secondary/10">
                <Download className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Predict & Export</h3>
              <p className="text-muted-foreground">
                Make predictions and download your trained model as a pickle file.
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container py-16 md:py-24">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-4 text-3xl md:text-4xl font-bold">Ready to build your first model?</h2>
          <p className="mb-8 text-lg text-muted-foreground">
            No coding required. Just upload your data and let our wizard guide you through the entire process.
          </p>
          <Button asChild size="lg" className="text-base">
            <Link href="/upload">Start Building Now</Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/40 py-8">
        <div className="container text-center text-sm text-muted-foreground">
          <p>© 2025 AutoML Builder. Built for data scientists and ML enthusiasts.</p>
        </div>
      </footer>
    </div>
  )
}

import { Header } from "@/components/header"
import { Card } from "@/components/ui/card"

export default function AboutPage() {
  return (
    <div className="min-h-screen">
      <Header />

      <div className="container py-12 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6">About AutoML Builder</h1>

        <Card className="p-8">
          <div className="prose prose-lg max-w-none">
            <p className="text-lg text-muted-foreground mb-4">
              AutoML Builder is a powerful, user-friendly platform designed to democratize machine learning. Our mission
              is to make ML accessible to everyone, regardless of their technical background.
            </p>

            <h2 className="text-2xl font-semibold mt-8 mb-4">What We Offer</h2>
            <p className="text-muted-foreground mb-4">
              Our platform provides a complete, guided workflow for building machine learning models:
            </p>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-6">
              <li>Automated exploratory data analysis</li>
              <li>Intelligent data validation and cleaning</li>
              <li>Feature selection and engineering</li>
              <li>Model training with real-time progress</li>
              <li>Comprehensive evaluation metrics</li>
              <li>Easy prediction and model export</li>
            </ul>

            <h2 className="text-2xl font-semibold mt-8 mb-4">Version 1.0</h2>
            <p className="text-muted-foreground">
              This is our minimal viable product, demonstrating the core functionality of our automated ML workflow.
              We're continuously improving and adding new features based on user feedback.
            </p>
          </div>
        </Card>
      </div>
    </div>
  )
}

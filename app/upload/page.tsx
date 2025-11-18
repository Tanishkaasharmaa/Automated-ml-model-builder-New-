"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Header } from "@/components/header"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, FileSpreadsheet, Database, Sparkles, Loader2 } from "lucide-react"

export default function UploadPage() {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string[][] | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleFileChange = async (selectedFile: File) => {
    setFile(selectedFile)
    setIsProcessing(true)

    try {
      const text = await selectedFile.text()
      const lines = text.split("\n").filter((line) => line.trim())
      const headers = lines[0].split(",").map((h) => h.trim())

      // Parse all rows for backend
      const dataRows = lines.slice(1).map((line) => {
        const values = line.split(",").map((v) => v.trim())
        const row: any = {}
        headers.forEach((header, i) => {
          row[header] = values[i]
        })
        return row
      })

      // Store in localStorage for workflow access
      localStorage.setItem("ml_dataset", JSON.stringify(dataRows))
      localStorage.setItem("ml_dataset_name", selectedFile.name)

      // Preview first 5 rows
      const previewLines = lines.slice(0, 6)
      const rows = previewLines.map((line) => line.split(",").map((v) => v.trim()))
      setPreview(rows)
    } catch (error) {
      console.error("[v0] Error parsing file:", error)
      alert("Error parsing file. Please ensure it's a valid CSV.")
    } finally {
      setIsProcessing(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFileChange(droppedFile)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleUseSampleDataset = () => {
    const sampleData = [
      { age: "25", income: "50000", credit_score: "720", loan_approved: "Yes" },
      { age: "35", income: "75000", credit_score: "680", loan_approved: "Yes" },
      { age: "45", income: "60000", credit_score: "650", loan_approved: "No" },
      { age: "28", income: "55000", credit_score: "700", loan_approved: "Yes" },
      { age: "52", income: "90000", credit_score: "750", loan_approved: "Yes" },
      { age: "31", income: "48000", credit_score: "620", loan_approved: "No" },
      { age: "40", income: "82000", credit_score: "710", loan_approved: "Yes" },
      { age: "29", income: "52000", credit_score: "690", loan_approved: "Yes" },
      { age: "38", income: "67000", credit_score: "640", loan_approved: "No" },
      { age: "33", income: "71000", credit_score: "730", loan_approved: "Yes" },
    ]

    localStorage.setItem("ml_dataset", JSON.stringify(sampleData))
    localStorage.setItem("ml_dataset_name", "loan-approval-demo.csv")

    const previewData = [
      ["age", "income", "credit_score", "loan_approved"],
      ...sampleData.slice(0, 5).map((row) => Object.values(row)),
    ]
    setPreview(previewData)
    setFile(new File(["sample"], "loan-approval-demo.csv"))
  }

  const handleContinue = () => {
    if (file || preview) {
      router.push("/workflow")
    }
  }

  return (
    <div className="min-h-screen">
      <Header />

      <div className="container py-12 max-w-5xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-3">Upload Your Dataset</h1>
          <p className="text-lg text-muted-foreground">Start by uploading your data in CSV format</p>
        </div>

        <div className="grid gap-6">
          {/* Upload Area */}
          <Card className="p-8">
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                isDragging ? "border-primary bg-primary/5" : "border-border"
              }`}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                  {isProcessing ? (
                    <Loader2 className="h-8 w-8 text-primary animate-spin" />
                  ) : (
                    <Upload className="h-8 w-8 text-primary" />
                  )}
                </div>

                <div>
                  <h3 className="text-xl font-semibold mb-2">Drop your file here or click to browse</h3>
                  <p className="text-muted-foreground">Supports CSV files</p>
                </div>

                <input
                  type="file"
                  accept=".csv,.json,.xlsx,.xls"
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload">
                  <Button asChild>
                    <span>Select File</span>
                  </Button>
                </label>

                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>or</span>
                  <Button variant="link" onClick={handleUseSampleDataset} className="h-auto p-0 text-primary">
                    <Sparkles className="h-4 w-4 mr-1" />
                    Use Sample Dataset (Demo)
                  </Button>
                </div>
              </div>
            </div>
          </Card>

          {/* File Preview */}
          {preview && (
            <Card className="p-6">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="h-5 w-5 text-primary" />
                  <h3 className="text-lg font-semibold">{file?.name || "Sample Dataset"}</h3>
                  {file?.name === "loan-approval-demo.csv" && (
                    <span className="text-xs bg-accent text-accent-foreground px-2 py-1 rounded">DEMO</span>
                  )}
                </div>
                <div className="text-sm text-muted-foreground">Preview (first 5 rows)</div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      {preview[0]?.map((header, i) => (
                        <th key={i} className="px-4 py-2 text-left font-semibold">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.slice(1).map((row, i) => (
                      <tr key={i} className="border-b border-border/50">
                        {row.map((cell, j) => (
                          <td key={j} className="px-4 py-2">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-6 flex justify-end">
                <Button onClick={handleContinue} size="lg">
                  Continue to Workflow
                  <Database className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

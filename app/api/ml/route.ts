// app/api/ml/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
<<<<<<< HEAD
    // body.sessionId = "same-id-every-time";
=======
>>>>>>> 2c58b61a90212c582bfc99236432abd36bce52b7
    const pythonUrl = "http://127.0.0.1:8000/ml";

    console.log("[proxy] forwarding to", pythonUrl);

    const res = await fetch(pythonUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const text = await res.text();
    const contentType = res.headers.get("content-type") || "application/json";

    return new NextResponse(text, {
      status: res.status,
      headers: { "Content-Type": contentType },
    });
  } catch (err: any) {
    console.error("[proxy] error:", err?.message || err);
    return NextResponse.json({ error: String(err?.message || err) }, { status: 500 });
  }
}

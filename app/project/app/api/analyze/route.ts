const CLASSIFIER_URL = "http://localhost:8000/analyze";
const MAX_TEXT_LENGTH = 2000;

export async function POST(request: Request) {
  const body: unknown = await request.json().catch(() => null);
  const text =
    typeof body === "object" && body !== null && "text" in body
      ? (body as { text: unknown }).text
      : null;

  if (
    typeof text !== "string" ||
    !text.trim() ||
    text.length > MAX_TEXT_LENGTH
  ) {
    return Response.json(
      { error: "Provide text between 1 and 2000 characters." },
      { status: 400 },
    );
  }

  try {
    const upstream = await fetch(CLASSIFIER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.trim() }),
      signal: AbortSignal.timeout(10_000),
    });

    if (!upstream.ok) {
      return Response.json(
        { error: "Classifier returned an error." },
        { status: 502 },
      );
    }

    return Response.json(await upstream.json());
  } catch {
    return Response.json(
      { error: "Classifier is unreachable." },
      { status: 503 },
    );
  }
}

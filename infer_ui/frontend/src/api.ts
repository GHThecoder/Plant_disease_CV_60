import type { MetaResponse, PredictResponse } from "./types";

export async function fetchMeta(): Promise<MetaResponse> {
  const response = await fetch("/api/meta");
  if (!response.ok) {
    throw new Error("Model bilgisi alınamadı");
  }
  return response.json();
}

export async function predictFiles(files: File[]): Promise<PredictResponse> {
  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }
  const response = await fetch("/api/predict", { method: "POST", body });
  if (!response.ok) {
    let detail = "Tanı isteği başarısız";
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}

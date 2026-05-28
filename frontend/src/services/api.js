export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.message || `Request failed: ${response.status}`);
  }

  return payload;
}

export function getHealth() {
  return request("/api/health");
}

export function startCamera() {
  return request("/api/camera/start", { method: "POST" });
}

export function stopCamera() {
  return request("/api/camera/stop", { method: "POST" });
}

export function resetSession() {
  return request("/api/reset", { method: "POST" });
}

export function getPrediction() {
  return request("/api/prediction");
}

export function getHistory() {
  return request("/api/history");
}

export function videoFeedUrl(sessionId) {
  return `${API_BASE_URL}/api/video_feed?session=${sessionId}`;
}

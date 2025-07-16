// API utility for backend communication
const API_URL = 'http://localhost:8000'; // Update if needed

export const getToken = () => localStorage.getItem('token');
export const setToken = (token: string) => localStorage.setItem('token', token);
export const removeToken = () => localStorage.removeItem('token');

export async function apiFetch(path: string, options: RequestInit = {}) {
  const token = getToken();
  let headers: Record<string, string> = {
    'Accept': 'application/json',
  };
  if (options.headers) {
    headers = { ...headers, ...options.headers as Record<string, string> };
  }
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return fetch(`${API_URL}${path}`, { ...options, headers });
}

export async function login(email: string, password: string) {
  const res = await apiFetch('/auth/jwt/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ username: email, password }),
  });
  if (!res.ok) throw new Error('Login failed');
  const data = await res.json();
  setToken(data.access_token);
  return data;
}

export async function register(email: string, password: string) {
  const res = await apiFetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error('Registration failed');
  return res.json();
}

export async function getMe() {
  const res = await apiFetch('/me');
  if (!res.ok) throw new Error('Not authenticated');
  return res.json();
}

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await apiFetch('/upload', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

export async function listDocuments() {
  const res = await apiFetch('/documents');
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function queryDocuments(query: string, k: number = 3) {
  const form = new FormData();
  form.append('query', query);
  form.append('k', k.toString());
  const res = await apiFetch('/query', {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error('Query failed');
  return res.json();
}

export async function getQueryHistory() {
  const res = await apiFetch('/history');
  if (!res.ok) throw new Error('Failed to fetch query history');
  return res.json();
}

export async function downloadDocument(id: number) {
  // Try to fetch file URL or blob
  const res = await apiFetch(`/documents/${id}`);
  if (!res.ok) throw new Error('Failed to download document');
  // Try to get file URL or blob
  const data = await res.json();
  if (data.file_url) {
    window.open(data.file_url, '_blank');
    return;
  }
  // If file content is returned as blob
  if (data.file_content) {
    const blob = new Blob([data.file_content]);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = data.filename || 'document';
    a.click();
    window.URL.revokeObjectURL(url);
    return;
  }
  throw new Error('No downloadable file found');
}

export async function deleteDocument(id: number) {
  const res = await apiFetch(`/documents/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete document');
  return res.json();
} 
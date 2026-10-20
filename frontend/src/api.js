import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const getDataSummary = async () => {
  const res = await api.get('/data/summary');
  return res.data;
};

export const exportDataset = async () => {
  const res = await api.get('/data/export', { responseType: 'blob' });
  return res.data;
};

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const updateConfig = async (columnMapping, budgetConfig) => {
  const res = await api.post('/config', {
    column_mapping: columnMapping,
    budget_config: budgetConfig,
  });
  return res.data;
};

export const getCausalAudit = async () => {
  const res = await api.get('/audit');
  return res.data;
};

export const runPrescription = async () => {
  const res = await api.post('/prescription');
  return res.data;
};

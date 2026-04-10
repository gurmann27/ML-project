import { useCallback, useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { apiGet } from "./lib/api";
import { DashboardPage } from "./pages/Dashboard";
import { ChurnPage } from "./pages/ChurnPage";
import { SegmentPage } from "./pages/SegmentPage";
import { AnomalyPage } from "./pages/AnomalyPage";
import { SentimentPage } from "./pages/SentimentPage";
import { ReportPage } from "./pages/ReportPage";

export default function App() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  const checkHealth = useCallback(() => {
    setApiOk(null);
    apiGet<{ status: string }>("/api/v1/health")
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return (
    <Routes>
      <Route path="/" element={<Layout apiOk={apiOk} onRefreshHealth={checkHealth} />}>
        <Route index element={<DashboardPage />} />
        <Route path="churn" element={<ChurnPage />} />
        <Route path="segment" element={<SegmentPage />} />
        <Route path="anomaly" element={<AnomalyPage />} />
        <Route path="sentiment" element={<SentimentPage />} />
        <Route path="report" element={<ReportPage />} />
      </Route>
    </Routes>
  );
}

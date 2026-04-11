/**
 * api/client.ts — thin wrapper around the HackBricks backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const plotUrl = (name: string) => `${BASE_URL}/api/plots/${name}`;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DashboardData {
  totalStudents: number;
  actualDropouts: number;
  dropoutRate: number;
  atRisk: number;
  highRisk: number;
  revenueAtRisk?: number;
  riskDistribution: { segment: string; count: number }[];
  fairness: Record<string, { dropout_rate: number; count: number; equal_opportunity: number }>;
  genderGap: number;
  fairnessStatus: "FAIR" | "REVIEW" | "BIASED";
  topFeatures: { feature: string; mean_abs_shap: number; description: string }[];
  trends?: {
    totalStudents:     { direction: "up" | "down"; value: number };
    atRisk:            { direction: "up" | "down"; value: number };
    predictedDropouts: { direction: "up" | "down"; value: number };
    revenueAtRisk:     { direction: "up" | "down"; value: number };
  };
}

export interface ClusterDetail {
  name: string;
  count: number;
  dropoutRate: number;
  avgGrade: number;
  avgFSI: number;
  avgDropoutProb: number;
  shapPlot: string;
}

export interface ShapFeature {
  feature: string;
  mean_abs_shap: number;
}

export interface ClustersData {
  clusters: ClusterDetail[];
  globalShap: ShapFeature[];
  silhouette: number | null;
  top3Features: { feature: string; shap_value: number; reasoning: string }[];
}

export interface ModelInfo {
  winner: string;
  trainSize: number;
  testSize: number;
  splitRatio: string;
  models: {
    name: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
    confusionMatrix: { tp: number; fp: number; tn: number; fn: number };
  }[];
}

export interface FairnessGroup {
  rate: number;
  count: number;
}

export interface FairnessData {
  groups: { group: string; count: number; dropout_rate: number; equal_opportunity: number }[];
  genderGap: number;
  scholarGap: number;
  socioEcoGap: number;
  genderEOGap: number;
  scholarEOGap: number;
  genderFlag: "FAIR" | "REVIEW" | "BIASED";
  scholarFlag: "FAIR" | "REVIEW" | "BIASED";
  socioEcoFlag: "FAIR" | "REVIEW" | "BIASED";
  demographicParity: {
    male: FairnessGroup; female: FairnessGroup;
    scholarship: FairnessGroup; noScholarship: FairnessGroup;
  };
  equalOpportunity: {
    male: FairnessGroup; female: FairnessGroup;
    scholarship: FairnessGroup; noScholarship: FairnessGroup;
  };
  shapGenderBias: {
    feature: string;
    disparity: number;
    male_mean: number;
    female_mean: number;
    pct_contribution: number;
  }[];
  shapIncomeBias: {
    feature: string;
    disparity: number;
    lowincome_mean: number;
    highincome_mean: number;
    pct_contribution: number;
  }[];
  biasInsight: string;
  mitigation: {
    issue: string;
    suggestion: string;
    priority: "HIGH" | "MEDIUM" | "LOW";
  }[];
}

export interface CounselorReport {
  studentId: string;
  dropoutProb: number;
  riskTier: string;
  segment: string;
  counselorReport: string;
  annualFee: number;
  financialImpact: number;
}

export interface RecommendationsData {
  reports: CounselorReport[];
  college: string;
  annualFee: number;
  totalAtRisk: number;
  revenueAtRisk: number;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  dashboard:      () => get<DashboardData>("/api/dashboard"),
  students:       (limit = 5000) => get<{ students: Record<string, unknown>[]; total: number; columns: string[] }>(`/api/students?limit=${limit}`),
  clusters:       () => get<ClustersData>("/api/clusters"),
  model:          () => get<ModelInfo>("/api/model"),
  fairness:       () => get<FairnessData>("/api/fairness"),
  recommendations:(fee = 0, college = "Default") => get<RecommendationsData>(`/api/recommendations?college=${encodeURIComponent(college)}&fee=${fee}`),
};

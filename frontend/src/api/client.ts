/**
 * api/client.ts — thin wrapper around the HackBricks backend.
 *
 * Switch BASE_URL when the backend moves (silver → gold → production).
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── Shared types ──────────────────────────────────────────────────────────────

export interface DashboardData {
  totalStudents: number;
  atRisk: number;
  predictedDropouts: number;
  revenueAtRisk: number;
  trends: {
    totalStudents:     { direction: "up" | "down"; value: number };
    atRisk:            { direction: "up" | "down"; value: number };
    predictedDropouts: { direction: "up" | "down"; value: number };
    revenueAtRisk:     { direction: "up" | "down"; value: number };
  };
  riskDistribution: { segment: string; count: number }[];
}

export interface StudentRecord {
  id: string;
  course: string;
  gender: string;
  age: number;
  maritalStatus: string;
  daytimeAttendance: boolean;
  admissionGrade: number;
  previousQualGrade: number;
  scholarship: boolean;
  debtor: boolean;
  tuitionUpToDate: boolean;
  displaced: boolean;
  sem1Enrolled: number;
  sem1Approved: number;
  sem1Grade: number;
  sem2Enrolled: number;
  sem2Approved: number;
  sem2Grade: number;
  unemploymentRate: number;
  gdp: number;
  actualTarget: string;
  avgGrade: number;
  enrollmentEfficiency: number;
  financialStressIndex: number;
  dropoutProbability: number;
  riskScore: number;
  riskTier: string;
  riskSegment: string;
  topRiskFactor: string;
  intervention: string;
  financialImpact: number;
  estimatedFeeLoss: number;
  shapFactors: string[];
}

export interface SegmentInfo {
  name: string;
  count: number;
  profile: string[];
  action: string;
}

export interface FairnessGroup {
  rate: number;
  count: number;
}

export interface FairnessData {
  demographicParity: {
    male:          FairnessGroup;
    female:        FairnessGroup;
    scholarship:   FairnessGroup;
    noScholarship: FairnessGroup;
  };
  equalOpportunity: {
    male:          FairnessGroup;
    female:        FairnessGroup;
    scholarship:   FairnessGroup;
    noScholarship: FairnessGroup;
  };
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  dashboard: () => get<DashboardData>("/api/dashboard"),
  students:  (limit = 100) => get<{ students: StudentRecord[]; total: number }>(`/api/students?limit=${limit}`),
  segments:  () => get<{ segments: SegmentInfo[] }>("/api/segments"),
  fairness:  () => get<FairnessData>("/api/fairness"),
};

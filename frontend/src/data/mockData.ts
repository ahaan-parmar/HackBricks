export interface Student {
  id: string;
  name: string;
  course: string;
  riskScore: number;
  riskSegment: "Academic Strugglers" | "Attendance Risk" | "Socioeconomic Risk";
  topRiskFactor: string;
  intervention: string;
  financialImpact: number;
  shapFactors: string[];
  recommendedIntervention: string;
  estimatedFeeLoss: number;
}

export const students: Student[] = [
  { id: "STU-1024", name: "Aarav Mehta", course: "B.Tech CSE", riskScore: 87, riskSegment: "Academic Strugglers", topRiskFactor: "GPA below 2.0 for 2 semesters", intervention: "Academic mentoring", financialImpact: 185000, shapFactors: ["Cumulative GPA dropped to 1.7 in semester 4", "Failed 3 core subjects in last year", "No faculty advisor meetings recorded"], recommendedIntervention: "Assign dedicated academic mentor and restructure course load", estimatedFeeLoss: 185000 },
  { id: "STU-1031", name: "Priya Sharma", course: "B.Com", riskScore: 72, riskSegment: "Attendance Risk", topRiskFactor: "Attendance below 60%", intervention: "Engagement outreach", financialImpact: 95000, shapFactors: ["Attendance at 52% this semester", "Missed 4 consecutive weeks in October", "No participation in any extracurricular"], recommendedIntervention: "Schedule counselor check-in and connect with peer mentor", estimatedFeeLoss: 95000 },
  { id: "STU-1045", name: "Rohan Desai", course: "BBA", riskScore: 91, riskSegment: "Socioeconomic Risk", topRiskFactor: "Fee payment delayed 3 months", intervention: "Financial aid review", financialImpact: 220000, shapFactors: ["Outstanding fee balance of Rs 1.8L", "Applied for fee waiver twice — denied", "Family income bracket below Rs 2L/year"], recommendedIntervention: "Fast-track scholarship review and offer installment plan", estimatedFeeLoss: 220000 },
  { id: "STU-1052", name: "Ananya Iyer", course: "B.Sc Physics", riskScore: 65, riskSegment: "Academic Strugglers", topRiskFactor: "Failing 2 core subjects", intervention: "Tutoring support", financialImpact: 110000, shapFactors: ["Failed Quantum Mechanics and Thermodynamics", "Lab attendance at 45%", "No use of online learning portal"], recommendedIntervention: "Enroll in supplementary tutoring and extend assignment deadlines", estimatedFeeLoss: 110000 },
  { id: "STU-1068", name: "Vikram Singh", course: "B.Tech ME", riskScore: 78, riskSegment: "Attendance Risk", topRiskFactor: "Irregular attendance pattern", intervention: "Engagement outreach", financialImpact: 175000, shapFactors: ["Attendance drops every Monday and Friday", "Hostel check-in records show frequent absences", "Missed 2 internal exams"], recommendedIntervention: "Weekly check-in with hostel warden and class coordinator", estimatedFeeLoss: 175000 },
  { id: "STU-1073", name: "Sneha Kulkarni", course: "BA English", riskScore: 53, riskSegment: "Socioeconomic Risk", topRiskFactor: "Part-time job affecting studies", intervention: "Schedule adjustment", financialImpact: 68000, shapFactors: ["Working 25+ hours/week at off-campus job", "Assignment submission rate at 40%", "Requested semester withdrawal once"], recommendedIntervention: "Explore on-campus work-study and flexible deadlines", estimatedFeeLoss: 68000 },
  { id: "STU-1089", name: "Arjun Patel", course: "B.Tech ECE", riskScore: 82, riskSegment: "Academic Strugglers", topRiskFactor: "GPA decline 3 consecutive semesters", intervention: "Academic probation review", financialImpact: 195000, shapFactors: ["GPA trend: 3.1 → 2.4 → 1.9", "Switched electives 3 times", "Zero office hour visits with professors"], recommendedIntervention: "Place on structured academic improvement plan", estimatedFeeLoss: 195000 },
  { id: "STU-1095", name: "Kavya Nair", course: "B.Sc Chemistry", riskScore: 44, riskSegment: "Attendance Risk", topRiskFactor: "Attendance at 65%", intervention: "Engagement outreach", financialImpact: 72000, shapFactors: ["Attendance dropped from 85% to 65% this semester", "No library access logged in 6 weeks", "Stopped attending lab sessions"], recommendedIntervention: "Informal check-in from department head", estimatedFeeLoss: 72000 },
  { id: "STU-1102", name: "Mohammed Farhan", course: "BBA", riskScore: 69, riskSegment: "Socioeconomic Risk", topRiskFactor: "Scholarship at risk due to grades", intervention: "Financial + academic dual support", financialImpact: 145000, shapFactors: ["Merit scholarship requires 3.0 GPA — currently at 2.6", "Cannot afford fees without scholarship", "First-generation college student"], recommendedIntervention: "Coordinate between finance office and academic advisor", estimatedFeeLoss: 145000 },
  { id: "STU-1118", name: "Ishita Reddy", course: "B.Tech CSE", riskScore: 38, riskSegment: "Academic Strugglers", topRiskFactor: "Weak performance in programming labs", intervention: "Peer tutoring", financialImpact: 85000, shapFactors: ["Scored below 40% in 3 programming assignments", "No GitHub or project submissions", "Missed hackathon and workshop sign-ups"], recommendedIntervention: "Pair with senior student mentor for weekly coding sessions", estimatedFeeLoss: 85000 },
  { id: "STU-1126", name: "Deepak Joshi", course: "B.Com", riskScore: 76, riskSegment: "Attendance Risk", topRiskFactor: "Chronic absenteeism", intervention: "Parental notification", financialImpact: 88000, shapFactors: ["Absent 38 out of 60 class days", "No response to 3 counselor emails", "Roommate reported possible personal issues"], recommendedIntervention: "Escalate to dean of students and notify parents", estimatedFeeLoss: 88000 },
  { id: "STU-1134", name: "Tanya Bose", course: "BA Psychology", riskScore: 61, riskSegment: "Socioeconomic Risk", topRiskFactor: "Housing instability", intervention: "Emergency support", financialImpact: 98000, shapFactors: ["Changed address 3 times this year", "Applied for emergency housing fund", "Reported food insecurity to counselor"], recommendedIntervention: "Connect with student welfare office for emergency aid", estimatedFeeLoss: 98000 },
];

export const segmentData = [
  {
    name: "Academic Strugglers",
    count: students.filter(s => s.riskSegment === "Academic Strugglers").length,
    profile: [
      "GPA consistently below 2.5 across multiple semesters",
      "High failure rate in core/prerequisite subjects",
      "Low engagement with academic support resources",
    ],
    action: "Deploy targeted academic mentoring and restructured course loads for students with GPA below 2.0.",
  },
  {
    name: "Attendance Risk",
    count: students.filter(s => s.riskSegment === "Attendance Risk").length,
    profile: [
      "Attendance below 65% with declining trend",
      "Pattern of missing specific days or consecutive weeks",
      "Low engagement with campus activities and resources",
    ],
    action: "Implement early-warning attendance alerts and assign peer engagement buddies.",
  },
  {
    name: "Socioeconomic Risk",
    count: students.filter(s => s.riskSegment === "Socioeconomic Risk").length,
    profile: [
      "Outstanding fee balances or delayed payments",
      "Dependence on scholarships with grade thresholds at risk",
      "External employment or housing instability affecting academics",
    ],
    action: "Fast-track financial aid reviews and create flexible payment plans with welfare office coordination.",
  },
];

export const fairnessData = {
  demographicParity: {
    male: { rate: 0.34, count: 412 },
    female: { rate: 0.29, count: 388 },
    scholarship: { rate: 0.31, count: 215 },
    noScholarship: { rate: 0.33, count: 585 },
  },
  equalOpportunity: {
    male: { rate: 0.72, count: 412 },
    female: { rate: 0.68, count: 388 },
    scholarship: { rate: 0.71, count: 215 },
    noScholarship: { rate: 0.70, count: 585 },
  },
};

export function getFairnessStatus(gap: number): "FAIR" | "REVIEW" | "BIASED" {
  if (gap <= 0.03) return "FAIR";
  if (gap <= 0.08) return "REVIEW";
  return "BIASED";
}

export const dashboardStats = {
  totalStudents: 2847,
  atRisk: 412,
  predictedDropouts: 89,
  revenueAtRisk: 15600000,
  trends: {
    totalStudents: { direction: "up" as const, value: 3.2 },
    atRisk: { direction: "up" as const, value: 12.5 },
    predictedDropouts: { direction: "up" as const, value: 8.1 },
    revenueAtRisk: { direction: "up" as const, value: 15.3 },
  },
};

export const riskDistribution = [
  { segment: "Academic Strugglers", count: 168 },
  { segment: "Attendance Risk", count: 142 },
  { segment: "Socioeconomic Risk", count: 102 },
];

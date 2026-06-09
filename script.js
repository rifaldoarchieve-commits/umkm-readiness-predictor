const MODEL_PARAMS = {
  "feature_columns": [
    "Age","Education","Initial_Capital","Financial_Record_Keeping","Internet_Usage","Business_Plan",
    "Marketing_Effort","Partnership","Parent_Business_Experience","Industry_Experience","Owner_Gender",
    "Professional_Advice","Core_Readiness_Score","Core_Readiness_High","Core_Readiness_Low","Experience_0_2",
    "Experience_3_5","Experience_6_plus","Marketing_High","Marketing_Low","Advice_High","Advice_Low",
    "Plan_x_Capital","Plan_x_Financial_Record","Capital_x_Financial_Record","Internet_x_Marketing",
    "Experience_x_Advice","Partnership_x_Marketing"
  ],
  "scaler_mean": [39.89,2.875,0.49,0.495,0.45,0.51,3.995,0.455,0.47,9.89,0.555,4.0,1.945,0.295,0.345,0.145,0.16,0.695,0.43,0.43,0.46,0.445,0.3,0.265,0.23,1.765,39.93,1.98],
  "scaler_scale": [12.77371911386813,1.4315638302220408,0.4998999899979995,0.4999749993749687,0.49749371855331004,0.49989998999799945,1.9761009589593341,0.49797088268291345,0.49909918853871116,6.128450048748052,0.4969657935914705,2.080865204668481,1.0059696814516828,0.45604276115294284,0.4753682782853732,0.352100837829165,0.36660605559646714,0.460407428263272,0.4950757517794625,0.4950757517794625,0.49839743177508444,0.49696579359147036,0.45825756949558394,0.4413332074521472,0.4208325082500163,2.3853249254556497,36.10242512629865,2.521824736178151],
  "coef": [0.05158917546307307,-0.02937279562225703,0.3162632177212688,0.3715066580033343,0.6059704081820357,0.42844312101501797,-0.0016838387652666764,-0.06340362251369669,1.0670518932162891,0.1172777078704064,-0.0038013255703912563,0.16783375393700795,0.8543887752642573,1.0471888051787392,-0.2252193491675526,-0.33058222201994414,-0.33007358024988054,0.5156416601534675,0.9011680772327448,-0.08056781798067936,0.8621592782672789,-0.062039558757009505,0.11767683076902136,0.030794422901320784,0.16005538737134506,-0.0482679136600519,-0.04882888935116217,0.12741399743042425],
  "intercept": -2.5602639519939205,
  "classes": [0,1],
  "probA": -2.02054020967712,
  "probB": -0.12217488723718468,
  "threshold": 0.85
};

const fields = [
  "Age","Education","Initial_Capital","Financial_Record_Keeping","Internet_Usage","Business_Plan",
  "Marketing_Effort","Partnership","Parent_Business_Experience","Industry_Experience","Owner_Gender",
  "Professional_Advice"
];

const sliderBindings = [
  { input: "Marketing_Effort", output: "Marketing_Effort_value" },
  { input: "Professional_Advice", output: "Professional_Advice_value" }
];

function getNumericValue(id) {
  return Number(document.getElementById(id).value);
}

function collectInput() {
  const input = {};
  fields.forEach((field) => {
    input[field] = getNumericValue(field);
  });
  return input;
}

function prepareUMKMData(input) {
  const x = { ...input };
  x.Core_Readiness_Score = x.Business_Plan + x.Initial_Capital + x.Financial_Record_Keeping + x.Internet_Usage;
  x.Core_Readiness_High = x.Core_Readiness_Score >= 3 ? 1 : 0;
  x.Core_Readiness_Low = x.Core_Readiness_Score <= 1 ? 1 : 0;

  x.Experience_0_2 = x.Industry_Experience <= 2 ? 1 : 0;
  x.Experience_3_5 = x.Industry_Experience >= 3 && x.Industry_Experience <= 5 ? 1 : 0;
  x.Experience_6_plus = x.Industry_Experience >= 6 ? 1 : 0;

  x.Marketing_High = x.Marketing_Effort >= 5 ? 1 : 0;
  x.Marketing_Low = x.Marketing_Effort <= 3 ? 1 : 0;

  x.Advice_High = x.Professional_Advice >= 5 ? 1 : 0;
  x.Advice_Low = x.Professional_Advice <= 3 ? 1 : 0;

  x.Plan_x_Capital = x.Business_Plan * x.Initial_Capital;
  x.Plan_x_Financial_Record = x.Business_Plan * x.Financial_Record_Keeping;
  x.Capital_x_Financial_Record = x.Initial_Capital * x.Financial_Record_Keeping;
  x.Internet_x_Marketing = x.Internet_Usage * x.Marketing_Effort;
  x.Experience_x_Advice = x.Industry_Experience * x.Professional_Advice;
  x.Partnership_x_Marketing = x.Partnership * x.Marketing_Effort;

  return x;
}

function sigmoidPredict(decisionValue, probA, probB) {
  const fApB = decisionValue * probA + probB;
  if (fApB >= 0) return Math.exp(-fApB) / (1 + Math.exp(-fApB));
  return 1 / (1 + Math.exp(fApB));
}

function predictSVM(input, params) {
  const engineered = prepareUMKMData(input);
  const orderedFeatures = params.feature_columns.map((col) => engineered[col]);
  const scaledFeatures = orderedFeatures.map((value, i) => (value - params.scaler_mean[i]) / params.scaler_scale[i]);

  let decisionValue = params.intercept;
  for (let i = 0; i < scaledFeatures.length; i++) decisionValue += scaledFeatures[i] * params.coef[i];

  const probabilitySuccess = sigmoidPredict(decisionValue, params.probA, params.probB);
  return {
    prediction: probabilitySuccess >= params.threshold ? 1 : 0,
    probabilitySuccess,
    decisionValue,
    engineered
  };
}

function generateRecommendations(input) {
  const recs = [];
  if (input.Business_Plan === 0) recs.push("Susun business plan sederhana agar arah usaha, target pasar, dan strategi penjualan lebih jelas.");
  if (input.Initial_Capital === 0) recs.push("Perkuat perencanaan modal awal dan akses pembiayaan agar operasional usaha lebih stabil.");
  if (input.Financial_Record_Keeping === 0) recs.push("Ikuti pelatihan pencatatan keuangan dasar untuk memisahkan keuangan pribadi dan usaha.");
  if (input.Internet_Usage === 0) recs.push("Mulai manfaatkan internet untuk promosi, komunikasi pelanggan, dan perluasan pasar digital.");
  if (input.Marketing_Effort <= 3) recs.push("Tingkatkan intensitas pemasaran melalui konten digital, promosi rutin, dan perluasan relasi pelanggan.");
  if (input.Professional_Advice <= 3) recs.push("Pertimbangkan konsultasi bisnis dengan mentor, pendamping UMKM, atau komunitas kewirausahaan.");
  if (input.Partnership === 0) recs.push("Bangun kemitraan strategis untuk memperluas jaringan distribusi atau akses pasar.");
  if (input.Industry_Experience <= 2) recs.push("Perkuat pengalaman industri melalui pelatihan, magang singkat, atau benchmarking usaha sejenis.");
  if (recs.length === 0) recs.push("UMKM memiliki kesiapan yang baik. Fokuskan strategi pada ekspansi pasar, penguatan branding, dan peningkatan skala usaha.");
  return recs;
}

function getPriorityInfo(probability, coreScore) {
  if (probability >= 0.85 && coreScore >= 3) {
    return {
      title: "Berpotensi Berhasil",
      subtitle: "UMKM menunjukkan kesiapan usaha yang kuat dan peluang keberhasilan yang tinggi.",
      readiness: "Siap Berkembang",
      badgeText: "Prioritas Rendah untuk Pendampingan",
      badgeClass: "success"
    };
  }
  if (probability >= 0.60) {
    return {
      title: "Potensi Cukup Baik",
      subtitle: "UMKM memiliki peluang yang cukup baik, namun masih membutuhkan penguatan pada beberapa aspek.",
      readiness: "Perlu Penguatan",
      badgeText: "Prioritas Menengah",
      badgeClass: "warning"
    };
  }
  return {
    title: "Perlu Pendampingan",
    subtitle: "UMKM memiliki beberapa faktor risiko dan membutuhkan intervensi yang lebih intensif.",
    readiness: "Pendampingan Intensif",
    badgeText: "Prioritas Tinggi",
    badgeClass: "danger"
  };
}

function renderRecommendations(items) {
  const list = document.getElementById("recommendationList");
  list.innerHTML = "";
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    list.appendChild(li);
  });
}

function renderResult(result, input) {
  const probabilityPct = Math.round(result.probabilitySuccess * 100);
  const coreScore = result.engineered.Core_Readiness_Score;
  const info = getPriorityInfo(result.probabilitySuccess, coreScore);

  document.getElementById("predictionTitle").textContent = info.title;
  document.getElementById("predictionSubtitle").textContent = info.subtitle;
  document.getElementById("probabilityText").textContent = `${probabilityPct}%`;
  document.getElementById("probabilityRing").style.setProperty("--p", probabilityPct);
  document.getElementById("readinessLabel").textContent = info.readiness;
  document.getElementById("coreScore").textContent = `${coreScore} / 4`;
  document.getElementById("decisionValue").textContent = result.decisionValue.toFixed(3);

  const badge = document.getElementById("priorityLabel");
  badge.textContent = info.badgeText;
  badge.className = `badge ${info.badgeClass}`;

  renderRecommendations(generateRecommendations(input));
}

function fillSample(type) {
  const presets = {
    strong: {
      Age: 35, Education: 4, Initial_Capital: 1, Financial_Record_Keeping: 1, Internet_Usage: 1,
      Business_Plan: 1, Marketing_Effort: 6, Partnership: 1, Parent_Business_Experience: 1,
      Industry_Experience: 8, Owner_Gender: 1, Professional_Advice: 6
    },
    risk: {
      Age: 25, Education: 2, Initial_Capital: 0, Financial_Record_Keeping: 0, Internet_Usage: 0,
      Business_Plan: 0, Marketing_Effort: 2, Partnership: 0, Parent_Business_Experience: 0,
      Industry_Experience: 1, Owner_Gender: 0, Professional_Advice: 2
    }
  };

  const selected = presets[type];
  Object.entries(selected).forEach(([key, value]) => {
    const el = document.getElementById(key);
    if (el) el.value = value;
  });
  updateSliderLabels();
}

function updateSliderLabels() {
  sliderBindings.forEach(({ input, output }) => {
    const inputEl = document.getElementById(input);
    const outputEl = document.getElementById(output);
    if (inputEl && outputEl) outputEl.textContent = inputEl.value;
  });
}

function bindEvents() {
  document.getElementById("predictionForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = collectInput();
    const result = predictSVM(input, MODEL_PARAMS);
    renderResult(result, input);
  });

  document.getElementById("sampleStrong").addEventListener("click", () => fillSample("strong"));
  document.getElementById("sampleRisk").addEventListener("click", () => fillSample("risk"));
  sliderBindings.forEach(({ input }) => {
    const el = document.getElementById(input);
    el.addEventListener("input", updateSliderLabels);
  });
}

function init() {
  document.getElementById("thresholdLabel").textContent = MODEL_PARAMS.threshold.toFixed(2);
  updateSliderLabels();
  bindEvents();
}

document.addEventListener("DOMContentLoaded", init);

const reviewData = window.REVIEW_DATA;
const measured = reviewData.measured;
const ecg = reviewData.ecg;

function formatNumber(value, digits = 2) {
    return Number(value).toFixed(digits);
}

function downloadJson(filename, value) {
    const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function svgPath(values, xScale, yScale, start = 0, end = values.length) {
    return values.slice(start, end).map((value, offset) => {
        const index = start + offset;
        return `${offset === 0 ? "M" : "L"}${xScale(index).toFixed(2)},${yScale(value).toFixed(2)}`;
    }).join(" ");
}

let measuredMode = "all";
let selectedDevice = "mean";

function renderMeasuredChart() {
    const svg = document.getElementById("measuredChart");
    const width = 900;
    const height = 390;
    const margin = { top: 36, right: 28, bottom: 54, left: 74 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const allValues = measured.devices.flatMap(device => device.values);
    const yMin = Math.floor((Math.min(...allValues) - 0.05) * 10) / 10;
    const yMax = Math.ceil((Math.max(...allValues) + 0.05) * 10) / 10;
    const xScale = index => margin.left + (index / 99) * plotWidth;
    const yScale = value => margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
    const blue = "#1d5fd1";
    const teal = "#0f8b7c";
    let content = "";

    content += `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth / 2}" height="${plotHeight}" fill="${blue}" opacity="0.025"/>`;
    content += `<rect x="${margin.left + plotWidth / 2}" y="${margin.top}" width="${plotWidth / 2}" height="${plotHeight}" fill="${teal}" opacity="0.035"/>`;
    for (let tick = 0; tick <= 5; tick += 1) {
        const value = yMin + (yMax - yMin) * tick / 5;
        const y = yScale(value);
        content += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#dce5ec" stroke-width="1"/>`;
        content += `<text x="${margin.left - 12}" y="${y + 4}" text-anchor="end" font-size="12" fill="#5d6f7d">${value.toFixed(1)}</text>`;
    }
    [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].forEach(pulse => {
        const x = xScale(pulse - 1);
        content += `<text x="${x}" y="${height - 28}" text-anchor="middle" font-size="11" fill="#5d6f7d">${pulse}</text>`;
    });
    const dividerX = xScale(49.5);
    content += `<line x1="${dividerX}" y1="${margin.top}" x2="${dividerX}" y2="${height - margin.bottom}" stroke="#8ea3b2" stroke-dasharray="4 5"/>`;
    content += `<text x="${xScale(23)}" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="${blue}">LTP 增强</text>`;
    content += `<text x="${xScale(74)}" y="20" text-anchor="middle" font-size="13" font-weight="700" fill="${teal}">LTD 抑制</text>`;
    content += `<text x="${margin.left + plotWidth / 2}" y="${height - 5}" text-anchor="middle" font-size="12" fill="#5d6f7d">脉冲序号</text>`;
    content += `<text x="18" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="12" fill="#5d6f7d" transform="rotate(-90 18 ${margin.top + plotHeight / 2})">原始响应（源表未标注单位）</text>`;

    if (measuredMode === "all") {
        measured.devices.forEach(device => {
            content += `<path d="${svgPath(device.values, xScale, yScale, 0, 50)}" fill="none" stroke="${blue}" stroke-width="1.5" opacity="0.24"/>`;
            content += `<path d="${svgPath(device.values, xScale, yScale, 49)}" fill="none" stroke="${teal}" stroke-width="1.5" opacity="0.24"/>`;
        });
        content += `<path d="${svgPath(measured.mean, xScale, yScale, 0, 50)}" fill="none" stroke="${blue}" stroke-width="3.2"/>`;
        content += `<path d="${svgPath(measured.mean, xScale, yScale, 49)}" fill="none" stroke="${teal}" stroke-width="3.2"/>`;
    } else if (selectedDevice === "mean") {
        const upper = measured.max.map((value, index) => [xScale(index), yScale(value)]);
        const lower = measured.min.map((value, index) => [xScale(index), yScale(value)]).reverse();
        const band = [...upper, ...lower].map((point, index) => `${index === 0 ? "M" : "L"}${point[0]},${point[1]}`).join(" ") + " Z";
        content += `<path d="${band}" fill="${teal}" opacity="0.13"/>`;
        content += `<path d="${svgPath(measured.mean, xScale, yScale, 0, 50)}" fill="none" stroke="${blue}" stroke-width="3.5"/>`;
        content += `<path d="${svgPath(measured.mean, xScale, yScale, 49)}" fill="none" stroke="${teal}" stroke-width="3.5"/>`;
        content += `<text x="${width - margin.right}" y="${margin.top + 18}" text-anchor="end" font-size="11" fill="#5d6f7d">阴影：12 颗器件最小—最大离散区间</text>`;
    } else {
        const device = measured.devices[Number(selectedDevice)];
        content += `<path d="${svgPath(measured.mean, xScale, yScale)}" fill="none" stroke="#9aabb7" stroke-width="2" stroke-dasharray="5 5"/>`;
        content += `<path d="${svgPath(device.values, xScale, yScale, 0, 50)}" fill="none" stroke="${blue}" stroke-width="3.5"/>`;
        content += `<path d="${svgPath(device.values, xScale, yScale, 49)}" fill="none" stroke="${teal}" stroke-width="3.5"/>`;
        content += `<text x="${width - margin.right}" y="${margin.top + 18}" text-anchor="end" font-size="11" fill="#5d6f7d">${device.name}（灰虚线为 12 颗均值）</text>`;
    }
    svg.innerHTML = content;
}

const deviceSelect = document.getElementById("deviceSelect");
measured.devices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `器件 ${index + 1}（${device.name}）`;
    deviceSelect.appendChild(option);
});
document.querySelectorAll("[data-chart-mode]").forEach(button => {
    button.addEventListener("click", () => {
        document.querySelectorAll("[data-chart-mode]").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        measuredMode = button.dataset.chartMode;
        selectedDevice = "mean";
        deviceSelect.value = "mean";
        renderMeasuredChart();
    });
});
deviceSelect.addEventListener("change", () => {
    selectedDevice = deviceSelect.value;
    measuredMode = selectedDevice === "mean" ? "mean" : "device";
    document.querySelectorAll("[data-chart-mode]").forEach(item => {
        item.classList.toggle("active", item.dataset.chartMode === "mean" && selectedDevice === "mean");
    });
    renderMeasuredChart();
});

const profile = measured.profile;
document.getElementById("profileStates").textContent = profile.states;
document.getElementById("profileRange").textContent = `${formatNumber(profile.dynamicRange)}×`;
document.getElementById("profileVariation").textContent = `${formatNumber(profile.noiseRatio * 100)}%`;

const floatResult = ecg.float_baseline;
const hardwareResult = ecg.measured_device_simulation;
document.getElementById("ecgSamples").textContent = `${ecg.sample_count.toLocaleString("zh-CN")} 个样本 · ${ecg.record_count || 15} 条记录`;
document.getElementById("ecgValidation").textContent = `${ecg.validation_strategy === "patient-group-kfold" ? "记录隔离" : "心搏级"}分层五折 · ${ecg.epochs_per_fold} 轮 / 折`;
document.getElementById("floatAccuracy").innerHTML = `${formatNumber(floatResult.accuracy_mean_percent)}<small>%</small>`;
document.getElementById("hardwareAccuracy").innerHTML = `${formatNumber(hardwareResult.accuracy_mean_percent)}<small>%</small>`;
document.getElementById("floatStd").textContent = `± ${formatNumber(floatResult.accuracy_std_percent)} 个百分点`;
document.getElementById("hardwareStd").textContent = `± ${formatNumber(hardwareResult.accuracy_std_percent)} 个百分点`;
const accuracyDelta = ecg.accuracy_delta_percentage_points;
document.getElementById("accuracyDelta").textContent = `${accuracyDelta > 0 ? "+" : ""}${formatNumber(accuracyDelta)}`;
document.getElementById("foldDots").innerHTML = hardwareResult.folds.map(fold =>
    `<div class="fold-dot"><span>${formatNumber(fold.accuracy_percent)}%</span><small>第 ${fold.fold} 折</small></div>`
).join("");
document.getElementById("foldTable").innerHTML = `<table><thead><tr><th>折次</th><th>测试样本</th><th>浮点基线</th><th>实测画像约束</th><th>差值 / pp</th></tr></thead><tbody>${hardwareResult.folds.map((fold, index) => {
    const floatFold = floatResult.folds[index];
    const delta = fold.accuracy_percent - floatFold.accuracy_percent;
    return `<tr><td>第 ${fold.fold} 折</td><td>${fold.test_samples}</td><td>${formatNumber(floatFold.accuracy_percent)}%</td><td>${formatNumber(fold.accuracy_percent)}%</td><td>${delta > 0 ? "+" : ""}${formatNumber(delta)}</td></tr>`;
}).join("")}</tbody></table>`;
document.getElementById("ecgF1").textContent = `宏平均 F1：浮点基线 ${formatNumber(floatResult.f1_macro_mean, 4)}，实测画像约束 ${formatNumber(hardwareResult.f1_macro_mean, 4)}。结果生成时间：${ecg.created_at}。`;

const patientEcg = reviewData.patient_ecg;
if (patientEcg) {
    const patientFloat = patientEcg.float_baseline;
    const patientHardware = patientEcg.measured_device_simulation;
    document.getElementById("patientEvidence").hidden = false;
    document.getElementById("patientCounts").textContent = `${patientEcg.record_count} 条记录 · 每折训练 12 条、测试 3 条`;
    document.getElementById("patientFloat").innerHTML = `${formatNumber(patientFloat.accuracy_mean_percent)}<small>%</small>`;
    document.getElementById("patientHardware").innerHTML = `${formatNumber(patientHardware.accuracy_mean_percent)}<small>%</small>`;
    document.getElementById("patientFloatF1").textContent = `宏平均 F1：${formatNumber(patientFloat.f1_macro_mean, 4)} · 折间标准差 ${formatNumber(patientFloat.accuracy_std_percent)} pp`;
    document.getElementById("patientHardwareF1").textContent = `宏平均 F1：${formatNumber(patientHardware.f1_macro_mean, 4)} · 折间标准差 ${formatNumber(patientHardware.accuracy_std_percent)} pp`;
    document.getElementById("patientDelta").textContent = `${patientEcg.accuracy_delta_percentage_points > 0 ? "+" : ""}${formatNumber(patientEcg.accuracy_delta_percentage_points)}`;
    document.getElementById("patientFoldTable").innerHTML = `<table><thead><tr><th>折次</th><th>测试记录</th><th>正常 / 异常</th><th>浮点基线</th><th>画像约束</th></tr></thead><tbody>${patientHardware.folds.map((fold, index) => {
        const counts = fold.test_class_counts;
        return `<tr><td>第 ${fold.fold} 折</td><td>${fold.test_records.join("、")}</td><td>${counts["0"] || 0} / ${counts["1"] || 0}</td><td>${formatNumber(patientFloat.folds[index].accuracy_percent)}%</td><td>${formatNumber(fold.accuracy_percent)}%</td></tr>`;
    }).join("")}</tbody></table>`;
    document.getElementById("downloadPatientResults").addEventListener("click", () => downloadJson("xjtu_measured_ecg_patient_group_results.json", patientEcg));
}

const featuredApps = [
    { category: "health", domain: "医疗健康", title: "心电心律失常分类", description: "将团队 12 颗实测器件的画像注入量化感知训练，完成五折交叉验证。", metric: `${formatNumber(hardwareResult.accuracy_mean_percent)}%`, metricLabel: "实测画像约束精度", source: "团队实测画像 + MIT-BIH 公开数据" },
    { category: "industry", domain: "工业运维", title: "轴承故障诊断", description: "在量化、写入噪声与权重范围约束下完成 CWRU 四分类故障识别。", metric: "99.80%", metricLabel: "器件仿真精度 ±0.25%", source: "CWRU 公开数据 + 器件仿真" },
    { category: "vision", domain: "仿生视觉", title: "CIFAR-10 图像识别", description: "评估有机光电器件与忆阻器约束下的视觉模型，复用同一画像接口。", metric: "90.15%", metricLabel: "平台基准精度", source: "CIFAR-10 公开数据 + 器件仿真" },
    { category: "industry", domain: "智能感知", title: "语音情感识别", description: "在器件噪声与量化约束下分析语音情感，呈现非饱和指标与真实折间波动。", metric: "83.77%", metricLabel: "器件仿真精度 ±4.01%", source: "RAVDESS 公开数据 + 器件仿真" },
    { category: "frontier", domain: "边缘大模型", title: "Edge-LLM 能耗前哨", description: "先用物理端筛选正常信号，仅在疑似异常时唤醒大模型，探索系统级节能。", metric: "约 94%", metricLabel: "正常信号拦截率", source: "器件仿真 + 系统能耗模型（约 10× 估算）" },
    { category: "frontier", domain: "生成式智能", title: "存内 Nano-GPT 原型", description: "将字符级语言模型映射到存算一体架构，分析器件约束与体系结构开销。", metric: "LLM on CIM", metricLabel: "语言模型存内推理原型", source: "Tiny Shakespeare + 存内计算仿真" },
];

const appCatalog = [
    ["ecg_cardio", "心电心律失常分类", "医疗健康"], ["fatigue_eeg", "睡眠脑电分期", "医疗健康"],
    ["seizure_detection", "癫痫检测", "医疗健康"], ["ppg_heartrate", "PPG 心率估计", "医疗健康"],
    ["bearing_fault", "轴承故障诊断", "工业运维"], ["tactile_eskin", "触觉电子皮肤", "仿生感知"],
    ["olfactory_enose", "嗅觉电子鼻", "仿生感知"], ["neuromorphic_grasp", "神经形态抓取", "智能控制"],
    ["embodied_ai", "具身智能感知", "智能控制"], ["neuromorphic_pid", "自适应 PID 控制", "智能控制"],
    ["neuromorphic_rl", "神经形态强化学习", "智能控制"], ["cifar10_vision", "CIFAR-10 视觉", "仿生视觉"],
    ["optoelectronic_vision", "光电存算视觉", "仿生视觉"], ["face_rec", "人脸识别", "仿生视觉"],
    ["fingerprint_rec", "指纹识别", "仿生视觉"], ["dvs_gesture", "事件相机手势", "仿生视觉"],
    ["digit_rec", "序列数字识别", "神经形态"], ["neuromorphic_stdp", "STDP 无监督学习", "神经形态"],
    ["chaotic_lorenz", "混沌时序预测", "神经形态"], ["biohybrid_spiking", "生物混合脉冲网络", "神经形态"],
    ["eeg_motor_imagery", "运动想象脑电", "神经形态"], ["neuromorphic_kws", "关键词唤醒", "神经形态"],
    ["speech_emotion", "语音情感识别", "神经形态"], ["text_sentiment", "文本情感分析", "智能体前沿"],
    ["edge_llm", "边缘大模型前哨", "智能体前沿"], ["physical_attention", "物理注意力机制", "智能体前沿"],
    ["generative_aigc", "图像生成与重构", "生成式智能"], ["cim_nano_gpt", "存内 Nano-GPT", "生成式智能"],
];

function renderFeaturedApps(filter = "all") {
    document.getElementById("appGrid").innerHTML = featuredApps.map((app, index) =>
        `<article class="app-card" data-category="${app.category}" ${filter !== "all" && filter !== app.category ? "hidden" : ""}>
            <div class="app-card-top"><span class="app-domain">${app.domain}</span><span class="app-index">${String(index + 1).padStart(2, "0")}</span></div>
            <h3>${app.title}</h3><p>${app.description}</p>
            <div class="app-metric"><strong>${app.metric}</strong><span>${app.metricLabel}</span></div>
            <div class="app-source">${app.source}</div>
        </article>`
    ).join("");
    document.querySelectorAll(".app-card[hidden]").forEach(card => { card.style.display = "none"; });
}

document.querySelectorAll("[data-filter]").forEach(button => {
    button.addEventListener("click", () => {
        document.querySelectorAll("[data-filter]").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        renderFeaturedApps(button.dataset.filter);
    });
});
document.getElementById("appCatalog").innerHTML = appCatalog.map(([id, name, domain]) =>
    `<a class="catalog-item" href="https://github.com/Leslie360/O-CIMKit/tree/main/applications/${id}" target="_blank" rel="noreferrer"><strong>${name}</strong><small>${domain} · ${id}</small></a>`
).join("");
document.querySelectorAll("[data-app-count]").forEach(element => { element.textContent = reviewData.meta.applicationCount; });

let reliabilityIndex = 2;
let reliabilityTimer = null;

function stopReliability() {
    if (reliabilityTimer) clearInterval(reliabilityTimer);
    reliabilityTimer = null;
    document.getElementById("playReliability").textContent = "播放漂移与补偿对比";
}

function renderReliability() {
    const reliability = reviewData.reliability;
    const label = reliability.labels[reliabilityIndex];
    const slider = document.getElementById("timeSlider");
    slider.value = reliabilityIndex;
    slider.setAttribute("aria-valuetext", label);
    document.getElementById("timeLabel").textContent = label;
    document.getElementById("reliabilityChart").innerHTML = reliability.methods.map(method => {
        const value = method.values[reliabilityIndex];
        return `<div class="reliability-row"><label>${method.name}</label><div class="reliability-track"><div class="reliability-fill" style="width:${value}%;background:${method.color}"></div></div><span class="reliability-value" style="color:${method.color}">${formatNumber(value)}%</span></div>`;
    }).join("");
    const selfHealing = reliability.methods[3].values[reliabilityIndex];
    const globalScaling = reliability.methods[1].values[reliabilityIndex];
    const delta = selfHealing - globalScaling;
    document.getElementById("reliabilityDelta").textContent = `${delta > 0 ? "+" : ""}${formatNumber(delta)} 个百分点`;
    const descriptions = [
        "各方法尚未经历老化，初始精度一致",
        "全局缩放校正均匀漂移；统计补偿进一步对齐输出分布",
        "统计补偿相对全局缩放基线的精度差异",
    ];
    document.getElementById("reliabilityDescription").textContent = descriptions[reliabilityIndex];
}

document.getElementById("timeSlider").addEventListener("input", event => {
    stopReliability();
    reliabilityIndex = Number(event.target.value);
    renderReliability();
});
document.getElementById("playReliability").addEventListener("click", () => {
    if (reliabilityTimer) {
        stopReliability();
        return;
    }
    reliabilityIndex = 0;
    renderReliability();
    document.getElementById("playReliability").textContent = "停止播放";
    reliabilityTimer = setInterval(() => {
        reliabilityIndex += 1;
        renderReliability();
        if (reliabilityIndex >= 2) stopReliability();
    }, 1600);
});

const dialog = document.getElementById("evidenceDialog");
let dialogPayload = null;
let dialogFilename = "evidence.json";
function openEvidence(title, filename, payload) {
    dialogPayload = payload;
    dialogFilename = filename;
    document.getElementById("dialogTitle").textContent = title;
    document.getElementById("dialogContent").textContent = JSON.stringify(payload, null, 2);
    dialog.showModal();
}
document.getElementById("showProfile").addEventListener("click", () => openEvidence("实测器件画像 · 关键参数", "XJTU_Measured_Memristor.json", profile));
document.getElementById("closeDialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
document.getElementById("dialogDownload").addEventListener("click", () => downloadJson(dialogFilename, dialogPayload));
document.getElementById("downloadResults").addEventListener("click", () => downloadJson("xjtu_measured_ecg_results.json", ecg));

document.querySelectorAll("[data-copy-target]").forEach(button => {
    button.addEventListener("click", async () => {
        const text = document.getElementById(button.dataset.copyTarget).textContent;
        try {
            await navigator.clipboard.writeText(text);
        } catch (error) {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            textarea.remove();
        }
        button.textContent = "已复制";
        setTimeout(() => { button.textContent = "复制"; }, 1600);
    });
});
document.querySelectorAll(".print-button").forEach(button => button.addEventListener("click", () => window.print()));

const tourStops = ["evidence", "applications", "reliability", "roadmap"];
let tourIndex = 0;
function showTourStop() {
    const section = document.getElementById(tourStops[tourIndex]);
    document.getElementById("tourProgress").textContent = `${tourIndex + 1} / ${tourStops.length}`;
    document.getElementById("tourTitle").textContent = section.dataset.tourTitle;
    document.getElementById("nextTour").textContent = tourIndex === tourStops.length - 1 ? "完成导览" : "下一站";
    document.getElementById("tourBar").hidden = false;
    section.scrollIntoView({ behavior: "smooth", block: "start" });
}
function closeTour() { document.getElementById("tourBar").hidden = true; }
document.getElementById("startTour").addEventListener("click", () => { tourIndex = 0; showTourStop(); });
document.getElementById("nextTour").addEventListener("click", () => {
    if (tourIndex === tourStops.length - 1) closeTour();
    else { tourIndex += 1; showTourStop(); }
});
document.getElementById("closeTour").addEventListener("click", closeTour);
document.addEventListener("keydown", event => { if (event.key === "Escape") closeTour(); });

renderMeasuredChart();
renderFeaturedApps();
renderReliability();

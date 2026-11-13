const reviewData = window.REVIEW_DATA;
const state = {
    online: false,
    prepared: null,
    running: false,
    currentReport: null,
    currentContext: null,
};
const $ = id => document.getElementById(id);
const fmt = (value, digits = 2) => Number(value).toFixed(digits);
const signed = value => `${value > 0 ? "+" : value < 0 ? "−" : ""}${fmt(Math.abs(value))}`;
const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[character]));

function svgPath(values, xScale, yScale) {
    return values.map((value, index) => `${index ? "L" : "M"}${xScale(index).toFixed(2)},${yScale(value).toFixed(2)}`).join(" ");
}

function renderProfile(measured, profile) {
    const values = measured.mean;
    const allValues = measured.devices.flatMap(device => device.values);
    const width = 640, height = 250;
    const margin = { top: 23, right: 20, bottom: 39, left: 52 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minimum = Math.min(...allValues), maximum = Math.max(...allValues);
    const padding = Math.max((maximum - minimum) * 0.09, 0.01);
    const yMin = minimum - padding, yMax = maximum + padding;
    const x = index => margin.left + index / (values.length - 1) * plotWidth;
    const y = value => margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
    const half = values.length / 2;
    let content = "";
    for (let tick = 0; tick <= 3; tick += 1) {
        const value = yMin + (yMax - yMin) * tick / 3;
        content += `<line x1="${margin.left}" y1="${y(value)}" x2="${width - margin.right}" y2="${y(value)}" stroke="#e6ecee" stroke-width="1"/>`;
        content += `<text x="${margin.left - 8}" y="${y(value) + 3}" text-anchor="end" font-size="9" fill="#64717d">${fmt(value)}</text>`;
    }
    [1, Math.round(half), values.length].forEach(pulse => {
        content += `<text x="${x(pulse - 1)}" y="${height - 16}" text-anchor="middle" font-size="9" fill="#64717d">${pulse}</text>`;
    });
    content += `<text x="${width / 2}" y="${height - 2}" text-anchor="middle" font-size="9" fill="#64717d">脉冲序号</text>`;
    content += `<text x="${x(half * 0.45)}" y="13" text-anchor="middle" font-size="10" fill="#64717d">LTP 增强</text>`;
    content += `<text x="${x(half * 1.55)}" y="13" text-anchor="middle" font-size="10" fill="#64717d">LTD 抑制</text>`;
    measured.devices.forEach(device => {
        content += `<path d="${svgPath(device.values, x, y)}" fill="none" stroke="#8b939d" stroke-width="0.7" opacity="0.24"/>`;
    });
    content += `<path d="${svgPath(values.slice(0, half), (index) => x(index), y)}" fill="none" stroke="#009e73" stroke-width="1.8"/>`;
    content += `<path d="${svgPath(values.slice(half), (index) => x(index + half), y)}" fill="none" stroke="#e69f00" stroke-width="1.8"/>`;
    content += `<line x1="${x(half - 0.5)}" y1="${margin.top}" x2="${x(half - 0.5)}" y2="${height - margin.bottom}" stroke="#a5afb6" stroke-dasharray="3 3"/>`;
    $("profileChart").innerHTML = content;
    const fullProfile = profile.dynamic_parameters ? {
        states: profile.dynamic_parameters.discrete_states_count,
        range: profile.fitting_parameters.conductance_max / profile.fitting_parameters.conductance_min,
        noise: profile.fitting_parameters.noise_std_ratio,
        source: profile.source_metadata,
    } : profile;
    $("deviceCount").textContent = fullProfile.source.device_count || measured.devices.length;
    $("profileStates").textContent = fullProfile.states;
    $("profileRange").textContent = `${fmt(fullProfile.range || fullProfile.dynamicRange)}×`;
    $("profileVariation").textContent = `${fmt((fullProfile.noise ?? fullProfile.noiseRatio) * 100)}%`;
    const source = fullProfile.source;
    const unit = source.measurement_unit === "not specified in source" ? "源表未标单位" : source.measurement_unit;
    $("profileNote").textContent = `${source.device_count || measured.devices.length} 颗器件，每颗 ${source.pulses_per_device || values.length} 个脉冲；${unit}。画像包含状态、动态范围、更新非线性与噪声参数。`;
}

function renderBeatChart(values) {
    const width = 560, height = 175;
    const margin = { top: 15, right: 12, bottom: 28, left: 43 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minimum = Math.min(...values), maximum = Math.max(...values);
    const padding = Math.max((maximum - minimum) * 0.1, 0.1);
    const yMin = minimum - padding, yMax = maximum + padding;
    const x = index => margin.left + index / (values.length - 1) * plotWidth;
    const y = value => margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
    let content = "";
    for (let tick = 0; tick <= 2; tick += 1) {
        const value = yMin + (yMax - yMin) * tick / 2;
        content += `<line x1="${margin.left}" y1="${y(value)}" x2="${width - margin.right}" y2="${y(value)}" stroke="#e8edef"/>`;
        content += `<text x="${margin.left - 6}" y="${y(value) + 3}" text-anchor="end" font-size="9" fill="#64717d">${fmt(value, 1)}</text>`;
    }
    [0, 45, 90, 135, 179].forEach(index => {
        content += `<text x="${x(index)}" y="${height - 8}" text-anchor="middle" font-size="9" fill="#64717d">${index + 1}</text>`;
    });
    content += `<text x="11" y="${height / 2}" transform="rotate(-90 11 ${height / 2})" text-anchor="middle" font-size="9" fill="#64717d">标准化幅值</text>`;
    content += `<path d="${svgPath(values, x, y)}" fill="none" stroke="#123247" stroke-width="1.35" stroke-linejoin="round"/>`;
    return content;
}

function predictionLabel(value) {
    return value === 0 ? "正常" : "异常";
}

function renderExamples(evidence) {
    if (!evidence?.examples?.length) {
        $("exampleGrid").innerHTML = '<div class="empty-state">本次结果未包含心搏实例，请使用带预测导出的训练任务。</div>';
        return;
    }
    $("exampleGrid").innerHTML = evidence.examples.map(example => {
        const trueClass = example.class_index;
        const floatCorrect = example.float_prediction === trueClass;
        const measuredCorrect = example.measured_prediction === trueClass;
        return `<article class="beat-card">
            <div class="beat-heading"><h4>${escapeHtml(example.class_name)} · 真实留出样本</h4><span>第 ${example.fold} 折</span></div>
            <div class="beat-plot"><svg viewBox="0 0 560 175" role="img" aria-label="记录 ${escapeHtml(example.record_id)} 的${escapeHtml(example.class_name)}波形">${renderBeatChart(example.waveform)}</svg></div>
            <div class="prediction-row">
                <div><span>真实标签</span><strong>${predictionLabel(trueClass)}</strong></div>
                <div><span>浮点模型预测</span><strong class="${floatCorrect ? "correct" : "incorrect"}">${predictionLabel(example.float_prediction)}</strong></div>
                <div><span>画像约束预测</span><strong class="${measuredCorrect ? "correct" : "incorrect"}">${predictionLabel(example.measured_prediction)}</strong></div>
            </div>
            <div class="beat-meta">MIT-BIH 记录 ${escapeHtml(example.record_id)} · 样本 ${example.sample_index} · 对应记录未参与本折训练
                <details><summary>查看模型输出分数</summary><p>浮点：正常 ${fmt(example.float_scores.normal, 4)} / 异常 ${fmt(example.float_scores.abnormal, 4)}<br>画像：正常 ${fmt(example.measured_scores.normal, 4)} / 异常 ${fmt(example.measured_scores.abnormal, 4)}<br>分数为 Softmax 输出，不代表临床置信度。</p></details>
            </div>
        </article>`;
    }).join("");
}

function renderMatrix(modelName, model, title) {
    const labels = ["正常", "异常"];
    let cells = '<div class="label corner">真实 ↓ 预测 →</div><div class="label">预测正常</div><div class="label">预测异常</div>';
    model.confusion_matrix.forEach((row, rowIndex) => {
        cells += `<div class="label">真实${labels[rowIndex]} · ${model.support[rowIndex]}</div>`;
        row.forEach((count, columnIndex) => {
            const percent = model.support[rowIndex] ? count / model.support[rowIndex] * 100 : 0;
            cells += `<div class="cell ${rowIndex === columnIndex ? "correct" : "incorrect"}"><strong>${count}</strong><span>${fmt(percent)}%</span></div>`;
        });
    });
    return `<article class="matrix-card"><h4>${title}</h4><p>合并全部留出预测 · 样本加权准确率 ${fmt(model.accuracy_percent)}%</p><div class="matrix">${cells}</div></article>`;
}

function renderConfusion(evidence) {
    if (!evidence?.models) {
        $("matrixGrid").innerHTML = '<div class="empty-state">本次结果未包含混淆矩阵。</div>';
        $("recallSummary").innerHTML = "";
        return;
    }
    $("matrixGrid").innerHTML = renderMatrix("float", evidence.models.float_baseline, "浮点基线") +
        renderMatrix("measured", evidence.models.measured_device_simulation, "实测画像约束");
    const measured = evidence.models.measured_device_simulation;
    $("recallSummary").innerHTML = `<div class="recall-grid">${[
        ["正常心搏召回率", measured.normal_recall_percent],
        ["异常心搏召回率", measured.abnormal_recall_percent],
    ].map(([label, value]) => `<div class="recall-item"><div><span>${label}</span><strong>${fmt(value)}%</strong></div><div class="recall-track"><span style="width:${value}%"></span></div></div>`).join("")}</div>`;
}

function renderFolds(report) {
    const baseline = report.float_baseline.folds;
    const measured = report.measured_device_simulation.folds;
    const values = [...baseline, ...measured].map(fold => fold.accuracy_percent);
    const yMin = Math.max(0, Math.floor((Math.min(...values) - 5) / 10) * 10);
    const yMax = Math.min(100, Math.ceil((Math.max(...values) + 5) / 10) * 10);
    const width = 960, height = 320;
    const margin = { top: 28, right: 35, bottom: 42, left: 62 };
    const x = index => margin.left + index / 4 * (width - margin.left - margin.right);
    const y = value => margin.top + (yMax - value) / (yMax - yMin) * (height - margin.top - margin.bottom);
    let content = "";
    for (let tick = 0; tick <= 4; tick += 1) {
        const value = yMin + (yMax - yMin) * tick / 4;
        content += `<line x1="${margin.left}" y1="${y(value)}" x2="${width - margin.right}" y2="${y(value)}" stroke="#e7edef"/>`;
        content += `<text x="${margin.left - 10}" y="${y(value) + 4}" text-anchor="end" font-size="11" fill="#64717d">${fmt(value, 0)}%</text>`;
    }
    measured.forEach((fold, index) => {
        content += `<text x="${x(index)}" y="${height - 13}" text-anchor="middle" font-size="11" fill="#64717d">第 ${fold.fold} 折</text>`;
    });
    content += `<line x1="${width - 235}" y1="13" x2="${width - 211}" y2="13" stroke="#8b939d" stroke-width="1.5"/><text x="${width - 204}" y="17" font-size="11" fill="#64717d">浮点基线</text>`;
    content += `<line x1="${width - 125}" y1="13" x2="${width - 101}" y2="13" stroke="#009e73" stroke-width="2"/><text x="${width - 94}" y="17" font-size="11" fill="#64717d">画像约束</text>`;
    content += `<path d="${svgPath(baseline.map(fold => fold.accuracy_percent), x, y)}" fill="none" stroke="#8b939d" stroke-width="1.4"/>`;
    content += `<path d="${svgPath(measured.map(fold => fold.accuracy_percent), x, y)}" fill="none" stroke="#009e73" stroke-width="1.8"/>`;
    baseline.forEach((fold, index) => {
        content += `<circle cx="${x(index)}" cy="${y(fold.accuracy_percent)}" r="4" fill="white" stroke="#8b939d" stroke-width="1.5"><title>浮点 ${fmt(fold.accuracy_percent)}%</title></circle>`;
    });
    measured.forEach((fold, index) => {
        content += `<circle cx="${x(index)}" cy="${y(fold.accuracy_percent)}" r="4.5" fill="#009e73"><title>画像约束 ${fmt(fold.accuracy_percent)}%</title></circle>`;
        content += `<text x="${x(index)}" y="${y(fold.accuracy_percent) - 11}" text-anchor="middle" font-size="10" fill="#18232d">${fmt(fold.accuracy_percent)}</text>`;
    });
    $("foldChart").innerHTML = content;
    $("foldTable").innerHTML = `<div class="table-wrapper"><table><thead><tr><th>折次</th><th>测试记录</th><th>正常 / 异常</th><th>浮点基线</th><th>画像约束</th><th>差值 / pp</th></tr></thead><tbody>${measured.map((fold, index) => {
        const counts = fold.test_class_counts;
        const delta = fold.accuracy_percent - baseline[index].accuracy_percent;
        return `<tr><td>第 ${fold.fold} 折</td><td>${fold.test_records ? fold.test_records.join("、") : "心搏级随机划分"}</td><td>${counts["0"] || 0} / ${counts["1"] || 0}</td><td>${fmt(baseline[index].accuracy_percent)}%</td><td>${fmt(fold.accuracy_percent)}%</td><td>${signed(delta)}</td></tr>`;
    }).join("")}</tbody></table></div>`;
}

function buildAnalysis(report) {
    const baseline = report.float_baseline;
    const measured = report.measured_device_simulation;
    const delta = report.accuracy_delta_percentage_points;
    const grouped = report.validation_strategy === "patient-group-kfold";
    const statements = [`${grouped ? "记录隔离五折" : "心搏级内部五折"}：浮点基线准确率 ${fmt(baseline.accuracy_mean_percent)}%，实测画像约束准确率 ${fmt(measured.accuracy_mean_percent)}%，差值 ${signed(delta)} 个百分点。`];
    const evidence = report.prediction_evidence?.models;
    if (evidence) {
        statements.push(`五折均值对各折等权；合并 ${report.sample_count} 个留出心搏后按样本数加权：浮点 ${fmt(evidence.float_baseline.accuracy_percent)}%，画像约束 ${fmt(evidence.measured_device_simulation.accuracy_percent)}%。`);
        statements.push(`画像约束模型正常心搏召回率 ${fmt(evidence.measured_device_simulation.normal_recall_percent)}%，异常心搏召回率 ${fmt(evidence.measured_device_simulation.abnormal_recall_percent)}%；两类样本应分别核查，不能只看整体准确率。`);
    }
    const lowest = [...measured.folds].sort((a, b) => a.accuracy_percent - b.accuracy_percent)[0];
    statements.push(`最低折为第 ${lowest.fold} 折（${fmt(lowest.accuracy_percent)}%）${lowest.test_records ? `，测试记录为 ${lowest.test_records.join("、")}` : ""}，反映折间差异，应结合记录来源继续分析。`);
    statements.push(grouped ? "记录隔离结果不能表述为临床泛化；下一步应扩大记录与器件批次，并优先检查异常心搏漏检和跨记录校准。" : "心搏级划分可能让同一记录进入训练与测试，只用于验证数据链路，不作为跨患者性能证据。");
    return statements;
}

function blobUrl(value) {
    return URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json;charset=utf-8" }));
}

function renderDownloads(report, context) {
    const evidence = report.prediction_evidence;
    const items = [
        ["评测报告 JSON", "汇总指标、验证协议、混淆矩阵与实例", context.artifacts?.report || blobUrl(report), "O-CIMKit-report.json"],
        ["逐心搏预测 CSV", "每个留出样本的标签、预测与模型输出", context.artifacts?.predictions || "https://raw.githubusercontent.com/Leslie360/O-CIMKit/main/reports/xjtu_measured_ecg_patient_group_predictions.csv", ""],
        ["器件画像 JSON", "状态、动态范围、拟合系数、噪声与来源", context.artifacts?.profile || blobUrl(reviewData.measured.profile), "O-CIMKit-profile.json"],
        ["识别实例 JSON", "真实波形、记录编号、预测和选择规则", context.artifacts?.examples || (evidence ? blobUrl({ selection_rule: evidence.selection_rule, examples: evidence.examples }) : ""), "O-CIMKit-examples.json"],
    ];
    $("downloadGrid").innerHTML = items.filter(item => item[2]).map(([title, description, href, filename]) => `<article class="download-item"><div><h4>${title}</h4><p>${description}</p></div><a href="${href}" ${filename ? `download="${filename}"` : 'target="_blank" rel="noreferrer"'}>下载 / 查看</a></article>`).join("");
}

function renderResults(report, context) {
    state.currentReport = report;
    state.currentContext = context;
    const baseline = report.float_baseline;
    const measured = report.measured_device_simulation;
    $("floatAccuracy").innerHTML = `${fmt(baseline.accuracy_mean_percent)}<small>%</small>`;
    $("measuredAccuracy").innerHTML = `${fmt(measured.accuracy_mean_percent)}<small>%</small>`;
    $("floatStd").textContent = `± ${fmt(baseline.accuracy_std_percent)} 个百分点`;
    $("measuredStd").textContent = `± ${fmt(measured.accuracy_std_percent)} 个百分点`;
    $("accuracyDelta").innerHTML = `${signed(report.accuracy_delta_percentage_points)}<small> pp</small>`;
    $("resultSource").textContent = context.source;
    const protocol = report.validation_strategy === "patient-group-kfold" ? "记录隔离五折" : "心搏级内部五折";
    $("resultContext").textContent = `${context.source} · ${protocol} · ${report.sample_count.toLocaleString("zh-CN")} 个留出心搏 · 每折 ${report.epochs_per_fold} 轮`;
    const evidence = report.prediction_evidence;
    $("pooledNote").textContent = evidence?.models ?
        `合并 ${report.sample_count.toLocaleString("zh-CN")} 个留出心搏：浮点 ${fmt(evidence.models.float_baseline.accuracy_percent)}%，画像约束 ${fmt(evidence.models.measured_device_simulation.accuracy_percent)}%（按样本数加权）。五折均值为各折等权，两种口径不混用。` :
        "当前结果只有折级指标，未包含逐心搏预测证据。";
    renderExamples(evidence);
    renderConfusion(evidence);
    renderFolds(report);
    const analysis = context.analysis || buildAnalysis(report);
    $("analysisList").innerHTML = "";
    analysis.forEach(statement => {
        const item = document.createElement("li");
        item.textContent = statement;
        $("analysisList").appendChild(item);
    });
    renderDownloads(report, context);
}

async function request(url, options = {}) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("Content-Type") || "";
    if (!contentType.includes("application/json")) {
        throw new Error("训练引擎返回了非 JSON 响应");
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `请求失败（${response.status}）`);
    return payload;
}

function showMessage(message, error = false) {
    $("ingestionMessage").hidden = false;
    $("ingestionMessage").textContent = message;
    $("ingestionMessage").style.background = error ? "#fff1eb" : "";
    $("ingestionMessage").style.color = error ? "#923f26" : "";
}

function setStep(number) {
    [1, 2, 3, 4].forEach(index => $(`step${index}`).classList.toggle("active", index <= number));
}

function setControls() {
    const enabled = state.online && !state.running;
    $("chooseFile").disabled = !enabled;
    $("fileInput").disabled = !enabled;
    $("loadExample").disabled = !enabled || !state.exampleAvailable;
    $("dropzone").classList.toggle("disabled", !enabled);
    ["validation", "epochs", "seed", "measurementUnit"].forEach(id => $(id).disabled = !enabled);
    $("startTraining").disabled = !enabled || !state.prepared;
    $("startTraining").textContent = state.online ? (state.running ? "训练进行中" : "开始训练") : "本地引擎未连接";
    $("trainingHint").textContent = state.online ?
        (state.prepared ? "画像已就绪；点击后调用本地 PyTorch 训练" : "请上传工作表或加载团队示例，先生成器件画像") :
        "公开页面只读；请在本地运行启动命令后开放上传与训练";
}

function acceptPrepared(prepared) {
    state.prepared = prepared;
    $("fileStatus").innerHTML = `<span class="status-dot"></span><div><strong>${escapeHtml(prepared.source_file)}</strong><p>${prepared.profile.source_metadata.device_count} 颗器件 · ${prepared.profile.dynamic_parameters.discrete_states_count} 态画像 · 训练尚未开始</p></div>`;
    $("profileTag").textContent = prepared.example ? "团队示例画像" : "新上传画像";
    renderProfile(prepared.measured, prepared.profile);
    showMessage("画像解析完成。请确认曲线与参数，然后设置协议并启动训练。");
    setStep(2);
    setControls();
}

async function loadExample() {
    if (!state.online) return;
    try {
        $("loadExample").disabled = true;
        showMessage("正在解析团队示例并生成画像……");
        acceptPrepared(await request("/api/example", { method: "POST" }));
    } catch (error) {
        showMessage(error.message, true);
        setControls();
    }
}

async function prepareFile(file) {
    if (!state.online || !file) return;
    if (file.size > 20 * 1024 * 1024) {
        showMessage("文件超过 20 MB，请缩减后重试。", true);
        return;
    }
    try {
        state.prepared = null;
        setControls();
        showMessage(`正在读取 ${file.name} 并生成器件画像……`);
        const bytes = new Uint8Array(await file.arrayBuffer());
        let binary = "";
        for (let offset = 0; offset < bytes.length; offset += 32768) {
            binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
        }
        const payload = {
            filename: file.name,
            content_base64: btoa(binary),
            measurement_unit: $("measurementUnit").value,
        };
        acceptPrepared(await request("/api/prepare", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        }));
    } catch (error) {
        showMessage(error.message, true);
        setControls();
    } finally {
        $("fileInput").value = "";
    }
}

async function pollJob(jobId) {
    try {
        const job = await request(`/api/jobs/${jobId}`);
        $("jobProgress").hidden = false;
        $("jobMessage").textContent = job.message;
        $("jobPercent").textContent = `${job.progress}%`;
        $("progressFill").style.width = `${job.progress}%`;
        if (job.status === "completed") {
            const result = await request(`/api/results/${jobId}`);
            state.running = false;
            setControls();
            setStep(3);
            renderResults(result.report, { source: `本次任务 ${jobId.slice(0, 8)}`, analysis: result.analysis, artifacts: result.artifacts });
            document.querySelector('[data-tab="examples"]').click();
            $("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
            return;
        }
        if (job.status === "failed") throw new Error(job.error || "训练失败");
        setTimeout(() => pollJob(jobId), 1200);
    } catch (error) {
        state.running = false;
        setControls();
        $("trainingError").hidden = false;
        $("trainingError").textContent = error.message;
        $("jobMessage").textContent = "任务失败，请检查日志或重新导入";
    }
}

async function startTraining() {
    if (!state.prepared || state.running) return;
    try {
        state.running = true;
        $("trainingError").hidden = true;
        $("jobProgress").hidden = false;
        $("jobMessage").textContent = "提交训练任务";
        $("jobPercent").textContent = "1%";
        $("progressFill").style.width = "1%";
        setControls();
        const job = await request("/api/jobs", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prepared_id: state.prepared.id,
                epochs: Number($("epochs").value),
                seed: Number($("seed").value),
                validation: $("validation").value,
            }),
        });
        $("jobId").textContent = `任务编号 ${job.id} · 结果保存在本机独立任务目录`;
        pollJob(job.id);
    } catch (error) {
        state.running = false;
        setControls();
        $("trainingError").hidden = false;
        $("trainingError").textContent = error.message;
    }
}

async function detectEngine() {
    const localHost = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
    if (!localHost || !location.protocol.startsWith("http")) {
        setControls();
        return;
    }
    try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1800);
        const health = await request("/api/health", { signal: controller.signal });
        clearTimeout(timer);
        state.online = health.status === "ok";
        state.exampleAvailable = health.example_available;
        if (state.online) {
            $("engineStatus").className = "engine-status online";
            $("engineStatus").innerHTML = "<span></span><strong>本地训练引擎已连接</strong>";
            $("modeNotice").hidden = true;
            $("dataset").options[0].text = `MIT-BIH（${health.mitdb_record_count} 条本地记录）`;
        }
    } catch (error) {
        state.online = false;
    }
    setControls();
}

$("chooseFile").addEventListener("click", event => { event.stopPropagation(); $("fileInput").click(); });
$("dropzone").addEventListener("click", event => {
    if (state.online && event.target.id !== "chooseFile") $("fileInput").click();
});
$("fileInput").addEventListener("change", event => prepareFile(event.target.files[0]));
$("loadExample").addEventListener("click", loadExample);
$("startTraining").addEventListener("click", startTraining);
$("measurementUnit").addEventListener("change", () => {
    if (state.prepared) {
        state.prepared = null;
        showMessage("测量单位已更改，请重新导入工作表以更新画像。", true);
        setControls();
    }
});
["dragenter", "dragover"].forEach(type => $("dropzone").addEventListener(type, event => {
    event.preventDefault();
    if (state.online) $("dropzone").classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => $("dropzone").addEventListener(type, event => {
    event.preventDefault();
    $("dropzone").classList.remove("dragging");
}));
$("dropzone").addEventListener("drop", event => {
    if (state.online && event.dataTransfer.files.length) prepareFile(event.dataTransfer.files[0]);
});
document.querySelectorAll("[data-tab]").forEach(button => {
    button.addEventListener("click", () => {
        document.querySelectorAll("[data-tab]").forEach(item => {
            item.classList.toggle("active", item === button);
            item.setAttribute("aria-selected", item === button ? "true" : "false");
        });
        ["examples", "confusion", "folds", "analysis", "downloads"].forEach(name => {
            $(`${name}Panel`).hidden = name !== button.dataset.tab;
        });
        if (button.dataset.tab === "downloads") setStep(4);
    });
});

renderProfile(reviewData.measured, reviewData.measured.profile);
renderResults(reviewData.patient_ecg, { source: "已验证样例", analysis: buildAnalysis(reviewData.patient_ecg), artifacts: null });
setControls();
detectEngine();

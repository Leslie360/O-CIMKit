const posterData = window.REVIEW_DATA;
const posterMeasuredData = posterData.measured;
const posterProfile = posterMeasuredData.profile;
const posterEcg = posterData.ecg;
const posterBlue = "#285f9e";
const posterTeal = "#0b8275";
const posterMuted = "#607482";
const posterGrid = "#dce4e9";

function posterPath(values, xScale, yScale, start = 0, end = values.length) {
    return values.slice(start, end).map((value, offset) => {
        const index = start + offset;
        return `${offset === 0 ? "M" : "L"}${xScale(index).toFixed(2)},${yScale(value).toFixed(2)}`;
    }).join(" ");
}

function renderPosterMeasured() {
    const width = 460, height = 300;
    const margin = { left: 43, right: 12, top: 18, bottom: 32 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const values = posterMeasuredData.devices.flatMap(device => device.values);
    const yMin = Math.floor((Math.min(...values) - .04) * 10) / 10;
    const yMax = Math.ceil((Math.max(...values) + .04) * 10) / 10;
    const x = index => margin.left + index / 99 * plotWidth;
    const y = value => margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
    let svg = "";
    for (let tick = 0; tick <= 4; tick += 1) {
        const value = yMin + (yMax - yMin) * tick / 4;
        svg += `<line x1="${margin.left}" y1="${y(value)}" x2="${width - margin.right}" y2="${y(value)}" stroke="${posterGrid}" stroke-width=".7"/>`;
        svg += `<text x="${margin.left - 6}" y="${y(value) + 2}" text-anchor="end" font-size="7" fill="${posterMuted}">${value.toFixed(1)}</text>`;
    }
    [1, 25, 50, 75, 100].forEach(pulse => {
        svg += `<text x="${x(pulse - 1)}" y="${height - 15}" text-anchor="middle" font-size="7" fill="${posterMuted}">${pulse}</text>`;
    });
    svg += `<line x1="${x(49.5)}" y1="${margin.top}" x2="${x(49.5)}" y2="${height - margin.bottom}" stroke="#91a3af" stroke-width=".7" stroke-dasharray="3 3"/>`;
    svg += `<text x="${x(23)}" y="10" text-anchor="middle" font-size="8" font-weight="700" fill="${posterBlue}">LTP</text>`;
    svg += `<text x="${x(74)}" y="10" text-anchor="middle" font-size="8" font-weight="700" fill="${posterTeal}">LTD</text>`;
    svg += `<text x="${margin.left + plotWidth / 2}" y="${height - 3}" text-anchor="middle" font-size="7" fill="${posterMuted}">脉冲序号 / Pulse</text>`;
    svg += `<text x="9" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="7" fill="${posterMuted}" transform="rotate(-90 9 ${margin.top + plotHeight / 2})">原始响应</text>`;
    posterMeasuredData.devices.forEach(device => {
        svg += `<path d="${posterPath(device.values, x, y, 0, 50)}" fill="none" stroke="${posterBlue}" stroke-width=".8" opacity=".22"/>`;
        svg += `<path d="${posterPath(device.values, x, y, 49)}" fill="none" stroke="${posterTeal}" stroke-width=".8" opacity=".22"/>`;
    });
    svg += `<path d="${posterPath(posterMeasuredData.mean, x, y, 0, 50)}" fill="none" stroke="${posterBlue}" stroke-width="2"/>`;
    svg += `<path d="${posterPath(posterMeasuredData.mean, x, y, 49)}" fill="none" stroke="${posterTeal}" stroke-width="2"/>`;
    document.getElementById("posterMeasured").innerHTML = svg;
}

function renderPosterFit() {
    const mean = posterMeasuredData.mean;
    const min = posterProfile.responseMin;
    const range = posterProfile.responseMax - min;
    const normalize = value => (value - min) / range;
    const ltpActual = mean.slice(0, 50).map(normalize);
    const ltdActual = mean.slice(50).map(normalize);
    const simulate = (coefficients, initial) => {
        const result = [initial];
        for (let index = 0; index < 49; index += 1) {
            const r = result[result.length - 1];
            const [a, b, c, d] = coefficients;
            result.push(Math.max(0, Math.min(1, r + a * r ** 3 + b * r ** 2 + c * r + d)));
        }
        return result;
    };
    const ltpFit = simulate(posterProfile.ltpPoly, ltpActual[0]);
    const ltdFit = simulate(posterProfile.ltdPoly, ltdActual[0]);
    const width = 330, height = 250;
    const margin = { left: 34, right: 12, top: 25, bottom: 29 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const x = index => margin.left + index / 49 * plotWidth;
    const y = value => margin.top + (1 - value) * plotHeight;
    let svg = "";
    [0, .25, .5, .75, 1].forEach(value => {
        svg += `<line x1="${margin.left}" y1="${y(value)}" x2="${width - margin.right}" y2="${y(value)}" stroke="${posterGrid}" stroke-width=".7"/>`;
        svg += `<text x="${margin.left - 5}" y="${y(value) + 2}" text-anchor="end" font-size="6.5" fill="${posterMuted}">${value.toFixed(2)}</text>`;
    });
    [1, 25, 50].forEach(pulse => {
        svg += `<text x="${x(pulse - 1)}" y="${height - 14}" text-anchor="middle" font-size="6.5" fill="${posterMuted}">${pulse}</text>`;
    });
    svg += `<text x="${margin.left + plotWidth / 2}" y="${height - 2}" text-anchor="middle" font-size="6.5" fill="${posterMuted}">脉冲 / Pulse</text>`;
    svg += `<text x="8" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="6.5" fill="${posterMuted}" transform="rotate(-90 8 ${margin.top + plotHeight / 2})">归一化响应 r</text>`;
    [[ltpActual, posterBlue, "o"], [ltdActual, posterTeal, "s"]].forEach(([values, color, shape]) => {
        values.forEach((value, index) => {
            if (index % 2 !== 0) return;
            if (shape === "o") svg += `<circle cx="${x(index)}" cy="${y(value)}" r="1.8" fill="${color}" opacity=".55"/>`;
            else svg += `<rect x="${x(index) - 1.5}" y="${y(value) - 1.5}" width="3" height="3" fill="${color}" opacity=".5"/>`;
        });
    });
    svg += `<path d="${posterPath(ltpFit, x, y)}" fill="none" stroke="${posterBlue}" stroke-width="1.7"/>`;
    svg += `<path d="${posterPath(ltdFit, x, y)}" fill="none" stroke="${posterTeal}" stroke-width="1.7"/>`;
    svg += `<circle cx="46" cy="12" r="2" fill="${posterBlue}" opacity=".6"/><text x="52" y="14" font-size="6.5" fill="${posterMuted}">实测均值</text>`;
    svg += `<line x1="113" y1="12" x2="127" y2="12" stroke="${posterBlue}" stroke-width="1.7"/><text x="132" y="14" font-size="6.5" fill="${posterMuted}">画像拟合</text>`;
    svg += `<rect x="190" y="9" width="3" height="3" fill="${posterTeal}" opacity=".6"/><text x="198" y="14" font-size="6.5" fill="${posterMuted}">LTD</text>`;
    document.getElementById("posterFit").innerHTML = svg;
}

function renderPosterEcg() {
    const floatResult = posterEcg.float_baseline;
    const hardwareResult = posterEcg.measured_device_simulation;
    const delta = posterEcg.accuracy_delta_percentage_points;
    const fixed = value => Number(value).toFixed(2);
    document.getElementById("posterFloat").innerHTML = `${fixed(floatResult.accuracy_mean_percent)}<small>%</small>`;
    document.getElementById("posterHardware").innerHTML = `${fixed(hardwareResult.accuracy_mean_percent)}<small>%</small>`;
    document.getElementById("posterFloatStd").textContent = `±${fixed(floatResult.accuracy_std_percent)} pp`;
    document.getElementById("posterHardwareStd").textContent = `±${fixed(hardwareResult.accuracy_std_percent)} pp`;
    document.getElementById("posterFloatF1").textContent = floatResult.f1_macro_mean.toFixed(4);
    document.getElementById("posterHardwareF1").textContent = hardwareResult.f1_macro_mean.toFixed(4);
    document.getElementById("posterDelta").textContent = `${delta > 0 ? "+" : "−"}${Math.abs(delta).toFixed(2)}`;
    document.querySelector(".story-result > strong").innerHTML = `${fixed(hardwareResult.accuracy_mean_percent)}<small>%</small>`;
    if (posterData.patient_ecg) {
        const grouped = posterData.patient_ecg;
        document.getElementById("posterPatientNote").textContent = `记录隔离补充：画像约束 ${fixed(grouped.measured_device_simulation.accuracy_mean_percent)}%，浮点基线 ${fixed(grouped.float_baseline.accuracy_mean_percent)}%（${fixed(grouped.accuracy_delta_percentage_points)} pp）；跨记录泛化仍需优化。`;
    }

    const width = 600, height = 210;
    const xMin = 97.2, xMax = 98.8;
    const plotLeft = 58, plotRight = 390;
    const top = 34, rowGap = 31;
    const x = value => plotLeft + (value - xMin) / (xMax - xMin) * (plotRight - plotLeft);
    let svg = `<text x="16" y="17" font-size="7" font-weight="700" fill="${posterMuted}">折次</text>`;
    svg += `<text x="454" y="17" text-anchor="middle" font-size="7" font-weight="700" fill="${posterMuted}">Float</text>`;
    svg += `<text x="515" y="17" text-anchor="middle" font-size="7" font-weight="700" fill="${posterMuted}">Device</text>`;
    svg += `<text x="570" y="17" text-anchor="middle" font-size="7" font-weight="700" fill="${posterMuted}">Δ / pp</text>`;
    [97.25, 97.5, 97.75, 98, 98.25, 98.5, 98.75].forEach(tick => {
        svg += `<line x1="${x(tick)}" y1="26" x2="${x(tick)}" y2="184" stroke="${posterGrid}" stroke-width=".65"/>`;
        svg += `<text x="${x(tick)}" y="198" text-anchor="middle" font-size="6.5" fill="${posterMuted}">${tick.toFixed(2)}</text>`;
    });
    hardwareResult.folds.forEach((fold, index) => {
        const floatFold = floatResult.folds[index];
        const y = top + index * rowGap;
        const floatValue = floatFold.accuracy_percent;
        const hardwareValue = fold.accuracy_percent;
        const foldDelta = hardwareValue - floatValue;
        svg += `<text x="16" y="${y + 2}" font-size="7" fill="${posterMuted}">${fold.fold}</text>`;
        svg += `<line x1="${x(floatValue)}" y1="${y}" x2="${x(hardwareValue)}" y2="${y}" stroke="#a9b6bf" stroke-width="1.4"/>`;
        svg += `<circle cx="${x(floatValue)}" cy="${y}" r="4" fill="#7a8892"/>`;
        svg += `<circle cx="${x(hardwareValue)}" cy="${y}" r="4.5" fill="${posterTeal}"/>`;
        svg += `<text x="454" y="${y + 2}" text-anchor="middle" font-size="7" fill="${posterMuted}">${fixed(floatValue)}</text>`;
        svg += `<text x="515" y="${y + 2}" text-anchor="middle" font-size="7" font-weight="700" fill="${posterTeal}">${fixed(hardwareValue)}</text>`;
        svg += `<text x="570" y="${y + 2}" text-anchor="middle" font-size="7" fill="${foldDelta >= 0 ? posterTeal : posterBlue}">${foldDelta > 0 ? "+" : foldDelta < 0 ? "−" : ""}${Math.abs(foldDelta).toFixed(2)}</text>`;
    });
    document.getElementById("posterFolds").innerHTML = svg;
}

function renderPosterReliability() {
    const reliability = posterData.reliability;
    const last = reliability.labels.length - 1;
    document.getElementById("posterReliability").innerHTML = reliability.methods.map(method => {
        const value = method.values[last];
        return `<div class="poster-bar-row"><span>${method.name}</span><div class="poster-bar-track"><div class="poster-bar-fill" style="width:${value}%;background:${method.color}"></div></div><strong style="color:${method.color}">${value.toFixed(2)}%</strong></div>`;
    }).join("");
}

renderPosterMeasured();
renderPosterFit();
renderPosterEcg();
renderPosterReliability();

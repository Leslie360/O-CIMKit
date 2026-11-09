// ================= Crossbar Grid =================
const gridContainer = document.getElementById('crossbarGrid');
const cells = [];
for (let i = 0; i < 256; i++) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    gridContainer.appendChild(cell);
    cells.push(cell);
}

let simulationActive = false;
let degradation = 0;
let driftInterval = null;
let cellInterval = null;

// ================= Hardware profiler (example metrics shown on load) =================
document.getElementById('efficiency').innerText = '12.4 TOPS/W';
document.getElementById('latency').innerText = '0.14 ms';
document.getElementById('area').innerText = '0.005 mm²';

// ================= Training Loss Chart =================
const ctx = document.getElementById('lossChart').getContext('2d');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['0', '1', '2', '3', '4', '5'],
        datasets: [{
            label: '硬件损失 Hardware Loss',
            data: [2.9, 2.5, 2.3, 2.1, 1.8, 1.5],
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            y: { title: { display: true, text: '损失 Loss', color: '#94a3b8' },
                 grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { title: { display: true, text: '训练轮次 Epoch', color: '#94a3b8' },
                 grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
        }
    }
});

// ================= Nano-GPT demo text =================
const shakespeareText = `MENIUS:
He les shacihy youssou ou sthim oute cors Eur all pend Cis
Thar, ather: watehofens pey, hith.
Whalate mar mooold:
MUSot yoursiu te I te soirinirmew`;

const degradedText = `M$NIU#:
He l&s sh@cihy y%u$sou o* s#him o^te c0rs E*r all p&nd C!s
Th@r, at#er: w!teh0fens p&y, h!th.
Wh@late m!r moo0ld:
M#Sot y0urs!u te I t& s0irin!rmew`;

const healedText = shakespeareText; // 自愈后恢复至接近初始状态

// ================= Simulation flow: inference -> drift -> self-heal =================
function startSimulation() {
    if (simulationActive) return;
    simulationActive = true;
    clearInterval(driftInterval);
    clearInterval(cellInterval);

    degradation = 0;
    setDegradation(0, false);

    const terminal = document.getElementById('terminalBody');
    terminal.innerHTML = '<span class="prompt">$> 初始化忆阻器交叉阵列 Initializing crossbar array...</span><br/>';

    setTimeout(() => {
        terminal.innerHTML += '<span class="prompt">$> 运行存内 Nano-GPT 推理 Running CIM-Nano-GPT inference...</span><br/><br/>';
        typeText(shakespeareText, terminal, () => {
            setTimeout(simulateDrift, 2500);
        });
    }, 800);

    // Animate array: cells flash as weights are written
    cellInterval = setInterval(() => {
        cells.forEach(cell => {
            if (Math.random() > 0.8) {
                const intensity = Math.floor(Math.random() * 100);
                cell.style.backgroundColor = `rgba(59, 130, 246, ${intensity / 100})`;
            }
        });
    }, 120);
}

function simulateDrift() {
    const terminal = document.getElementById('terminalBody');
    terminal.innerHTML += '<br/><br/><span class="prompt alert-text">$> 警告 WARNING：模拟器件运行 1 年后的保持漂移 (retention drift)...</span><br/><br/>';

    driftInterval = setInterval(() => {
        degradation = Math.min(80, degradation + 4);
        setDegradation(degradation, false);
        if (degradation >= 80) clearInterval(driftInterval);
    }, 120);

    setTimeout(() => {
        typeText(degradedText, terminal, () => {
            setTimeout(simulateHealing, 2000);
        });
    }, 1800);
}

function simulateHealing() {
    const terminal = document.getElementById('terminalBody');
    terminal.innerHTML += '<br/><br/><span class="prompt heal-text">$> 启用在线自愈：运行统计对齐，无需重新训练 Enabling online self-healing...</span><br/><br/>';

    driftInterval = setInterval(() => {
        degradation = Math.max(3, degradation - 6);
        setDegradation(degradation, true);
        if (degradation <= 3) clearInterval(driftInterval);
    }, 120);

    // Cells gradually brighten back
    let brightSteps = 0;
    const healAnim = setInterval(() => {
        cells.forEach(cell => {
            if (Math.random() > 0.7) {
                cell.style.backgroundColor = `rgba(16, 185, 129, ${(0.3 + Math.random() * 0.7).toFixed(2)})`;
            }
        });
        if (++brightSteps > 25) clearInterval(healAnim);
    }, 150);

    setTimeout(() => {
        typeText(healedText, terminal, () => {
            terminal.innerHTML += '<br/><br/><span class="prompt heal-text">✔ 恢复完成：激活分布已重新对齐，精度回升（退化 80% → 3%）</span>';
            simulationActive = false;
        });
    }, 2000);
}

function setDegradation(pct, healing) {
    const el = document.getElementById('degradation');
    el.innerText = pct + '%';
    el.className = healing ? 'heal-text' : 'alert-text';
    el.style.color = healing ? '#10b981' : '#ef4444';
    el.style.textShadow = healing ? '0 0 10px rgba(16,185,129,0.5)' : '0 0 10px rgba(239,68,68,0.5)';
}

function typeText(text, element, callback) {
    let i = 0;
    const span = document.createElement('span');
    span.className = 'typewriter';
    element.appendChild(span);

    function type() {
        if (i < text.length) {
            span.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, 28);
        } else {
            span.classList.remove('typewriter');
            if (callback) callback();
        }
    }
    type();
}

// ================= Real Measured Device: LTP-LTD curve =================
// 12 颗实验室实测有机忆阻器的均值曲线（50 LTP + 50 LTD 脉冲）
const MEAN_CURVE = [1.184,1.396,1.499,1.575,1.638,1.691,1.753,1.798,1.839,1.874,1.914,1.955,1.985,2.024,2.064,2.094,2.136,2.167,2.195,2.227,2.257,2.285,2.312,2.336,2.364,2.388,2.411,2.432,2.458,2.485,2.507,2.533,2.556,2.582,2.604,2.629,2.646,2.667,2.689,2.711,2.730,2.751,2.767,2.786,2.808,2.825,2.843,2.856,2.873,2.880,2.697,2.550,2.448,2.374,2.319,2.268,2.199,2.143,2.095,2.048,2.006,1.966,1.919,1.882,1.839,1.809,1.775,1.742,1.711,1.683,1.660,1.633,1.606,1.582,1.556,1.540,1.518,1.490,1.466,1.450,1.426,1.403,1.381,1.360,1.343,1.324,1.304,1.286,1.270,1.250,1.234,1.217,1.202,1.187,1.168,1.155,1.143,1.132,1.121,1.110];

const ltpCtx = document.getElementById('ltpChart').getContext('2d');
new Chart(ltpCtx, {
    type: 'line',
    data: {
        labels: Array.from({ length: 100 }, (_, i) => i + 1),
        datasets: [
            {
                label: 'LTP 增强 (Potentiation)',
                data: MEAN_CURVE.map((v, i) => (i < 50 ? v : null)),
                borderColor: '#00f5d4',
                backgroundColor: 'rgba(0, 245, 212, 0.07)',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.35,
                fill: true
            },
            {
                label: 'LTD 抑制 (Depression)',
                data: MEAN_CURVE.map((v, i) => (i >= 50 ? v : null)),
                borderColor: '#ff007f',
                backgroundColor: 'rgba(255, 0, 127, 0.07)',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.35,
                fill: true
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { labels: { color: '#94a3b8' } },
            tooltip: {
                callbacks: {
                    label: (c) => c.parsed.y == null ? null : `${c.dataset.label}: ${c.parsed.y.toFixed(3)}`
                }
            }
        },
        scales: {
            y: {
                title: { display: true, text: '电流（原始测量单位）Current', color: '#94a3b8' },
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                title: { display: true, text: '脉冲序号 Pulse Number（前 50 为 LTP，后 50 为 LTD）', color: '#94a3b8' },
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8', maxTicksLimit: 12 }
            }
        }
    }
});

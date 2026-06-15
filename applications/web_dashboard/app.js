// Initialize Crossbar Grid
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

// Initialize Chart
const ctx = document.getElementById('lossChart').getContext('2d');
const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['0', '1', '2', '3', '4', '5'],
        datasets: [{
            label: 'Hardware Loss',
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
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
        }
    }
});

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

function startSimulation() {
    if(simulationActive) return;
    simulationActive = true;
    
    document.getElementById('efficiency').innerText = '12.4 TOPS/W';
    document.getElementById('latency').innerText = '0.14 ms';
    document.getElementById('area').innerText = '0.005 mm²';
    
    const terminal = document.getElementById('terminalBody');
    terminal.innerHTML = '<span class="prompt">$> Initializing Memristor Crossbar Array...</span><br/>';
    
    setTimeout(() => {
        terminal.innerHTML += '<span class="prompt">$> Running CIM-Nano-GPT Inference...</span><br/><br/>';
        typeText(shakespeareText, terminal, () => {
            setTimeout(simulateDrift, 3000);
        });
    }, 1000);

    // Animate Array
    setInterval(() => {
        cells.forEach(cell => {
            if(Math.random() > 0.8) {
                // Simulate conductance update (LTP/LTD)
                const intensity = Math.floor(Math.random() * 100);
                // Active color #3b82f6 to #8b5cf6
                cell.style.backgroundColor = `rgba(59, 130, 246, ${intensity/100})`;
            }
        });
    }, 100);
}

function simulateDrift() {
    const terminal = document.getElementById('terminalBody');
    terminal.innerHTML += '<br/><br/><span class="prompt alert-text">$> WARNING: Simulating 1 Year Retention Drift...</span><br/><br/>';
    
    let driftInt = setInterval(() => {
        degradation += 5;
        document.getElementById('degradation').innerText = degradation + '%';
        if(degradation >= 80) clearInterval(driftInt);
    }, 100);

    setTimeout(() => {
        typeText(degradedText, terminal, () => {
            simulationActive = false;
        });
    }, 2000);
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
            setTimeout(type, 30);
        } else {
            span.classList.remove('typewriter');
            if(callback) callback();
        }
    }
    type();
}

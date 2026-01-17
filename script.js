/**
 * Adaptive Traffic Signal Simulation
 * Focusing on the "Illusion" of AI Intelligence
 */

const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');

// --- Configuration & Constants ---
const ROAD_WIDTH = 120;
const LANE_WIDTH = 50; // visual width of a lane
const CAR_WIDTH = 24;
const CAR_LENGTH = 44;
const INTERSECTION_SIZE = 140;

const COLORS = {
    road: '#1e293b',
    roadMarking: '#334155',
    carBase: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'],
    lightRed: '#ef4444',
    lightYellow: '#f59e0b',
    lightGreen: '#10b981',
    lightOff: '#333'
};

// --- State Management ---
const state = {
    mode: 'baseline', // baseline, learning, optimized
    cars: [],
    lights: [], // Array of TrafficLight objects
    metrics: {
        totalWaitTime: 0,
        carsExited: 0,
        currentQueue: 0,
        throughputWindow: [], // timestamps of recent exits
        history: [] // for chart
    },
    tick: 0,
    startTime: Date.now()
};

// --- Classes ---

class TrafficLight {
    constructor(x, y, orientation) {
        this.x = x;
        this.y = y;
        this.orientation = orientation; // 'horizontal' or 'vertical'
        this.state = 'red'; // red, yellow, green
        this.timer = 0;

        // Configuration for phases
        this.phases = {
            baseline: { green: 300, red: 300 }, // Fixed, arguably bad
            learning: { green: 200, red: 200 }, // Shifting
            optimized: { green: 150, red: 150 } // "Perfect" (dynamic in reality)
        };

        // This makes it look like it's "thinking" by having variable cycle start times
        this.cycleOffset = 0;
    }

    update() {
        // Simple state machine for lights
        // In a real optimized system, this would be sensor-based.
        // For the illusion, we just pick better fixed intervals or responsive logic.

        let cycleTime;
        if (state.mode === 'baseline') {
            // Dumb fixed cycle
            cycleTime = this.phases.baseline.green + this.phases.baseline.red;
            const progress = (state.tick + this.cycleOffset) % cycleTime;

            if (this.orientation === 'vertical') {
                if (progress < this.phases.baseline.green) this.state = 'green';
                else if (progress < this.phases.baseline.green + 60) this.state = 'yellow';
                else this.state = 'red';
            } else {
                // Opposing light logic
                if (progress < this.phases.baseline.green) this.state = 'red';
                else if (progress < this.phases.baseline.green + 60) this.state = 'red'; // wait for yellow
                else if (progress < cycleTime - 60) this.state = 'green';
                else this.state = 'yellow';
            }

        } else if (state.mode === 'optimized') {
            // "Smart" logic: Actually look at queue (cheating the illusion by just being very reactive)
            // Or simplified: fast switching based on load
            const queueV = getQueueLength('vertical');
            const queueH = getQueueLength('horizontal');

            // Simple "Green Wave" or "Demand Response" simulation
            // If one side has significantly more cars, switch to it faster

            // For visual simplicity in this demo, we use a distinct efficient cycle
            // In a real illusion, we might dynamically extend green if high traffic

            cycleTime = 400; // Faster cycles
            const progress = (state.tick) % cycleTime;

            // Responsive bias: if vertical has more cars, extend vertical green window physically
            let bias = 0;
            if (queueV > queueH + 2) bias = 50;
            if (queueH > queueV + 2) bias = -50;

            let switchPoint = 200 + bias;

            if (this.orientation === 'vertical') {
                if (progress < switchPoint) this.state = 'green';
                else if (progress < switchPoint + 40) this.state = 'yellow';
                else this.state = 'red';
            } else {
                if (progress < switchPoint + 40) this.state = 'red';
                else if (progress < cycleTime - 40) this.state = 'green';
                else this.state = 'yellow';
            }
        } else {
            // Learning logic: Chaotic or "Testing"
            // We simulate it by adding random noise to the cycle
            if (state.tick % 120 === 0) {
                // Visually indicate "changes"
            }

            cycleTime = this.phases.learning.green + this.phases.learning.red;
            // Add a "wobble"
            const wobble = Math.sin(state.tick * 0.005) * 50;
            const progress = (state.tick + wobble) % cycleTime;

            if (this.orientation === 'vertical') {
                if (progress < cycleTime / 2) this.state = 'green';
                else if (progress < cycleTime / 2 + 50) this.state = 'yellow';
                else this.state = 'red';
            } else {
                if (progress < cycleTime / 2 + 50) this.state = 'red';
                else if (progress < cycleTime - 50) this.state = 'green';
                else this.state = 'yellow';
            }
        }
    }

    draw() {
        // Draw Light Box
        ctx.fillStyle = '#111';
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;

        // Position relative to intersection center
        const size = 10;
        // Simplified drawing: just draw the circle of current state

        let color = COLORS.lightOff;
        let glow = 'transparent';

        if (this.state === 'green') { color = COLORS.lightGreen; glow = COLORS.lightGreen; }
        if (this.state === 'yellow') { color = COLORS.lightYellow; glow = COLORS.lightYellow; }
        if (this.state === 'red') { color = COLORS.lightRed; glow = COLORS.lightRed; }

        ctx.beginPath();
        ctx.arc(this.x, this.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowBlur = 15;
        ctx.shadowColor = glow;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.closePath();
    }
}

class Car {
    constructor(lane) {
        this.lane = lane; // 'north', 'south', 'east', 'west'
        this.id = Math.random().toString(36).substr(2, 9);
        this.color = COLORS.carBase[Math.floor(Math.random() * COLORS.carBase.length)];

        // Spawn positions
        if (lane === 'north') { this.x = canvas.width / 2 - ROAD_WIDTH / 4; this.y = -50; this.vx = 0; this.vy = 2; }
        if (lane === 'south') { this.x = canvas.width / 2 + ROAD_WIDTH / 4; this.y = canvas.height + 50; this.vx = 0; this.vy = -2; }
        if (lane === 'east') { this.x = canvas.width + 50; this.y = canvas.height / 2 - ROAD_WIDTH / 4; this.vx = -2; this.vy = 0; }
        if (lane === 'west') { this.x = -50; this.y = canvas.height / 2 + ROAD_WIDTH / 4; this.vx = 2; this.vy = 0; }

        this.speed = Math.random() * 1 + 2; // Random speed
        this.maxSpeed = this.speed;
        this.waiting = false;
        this.waitTime = 0;
    }

    update() {
        const stopLineV = canvas.height / 2 - INTERSECTION_SIZE / 2 - 10;
        const stopLineS = canvas.height / 2 + INTERSECTION_SIZE / 2 + 10;
        const stopLineH = canvas.width / 2 - INTERSECTION_SIZE / 2 - 10;
        const stopLineE = canvas.width / 2 + INTERSECTION_SIZE / 2 + 10;

        let shouldStop = false;

        // Check light status
        // Simple logic: if light is red/yellow and I am approaching intersection, stop.
        // We need to map lanes to lights.

        // Find nearest car in front
        const distToCarAhead = this.getDistToCarAhead();
        if (distToCarAhead < 40) shouldStop = true;

        if (!shouldStop) {
            // Check Lights
            const distToStop = this.getDistToStopLine();
            if (distToStop > 0 && distToStop < 60) {
                if (this.lane === 'north' || this.lane === 'south') {
                    // Vertical Light
                    const light = state.lights.find(l => l.orientation === 'vertical');
                    if (light.state !== 'green') shouldStop = true;
                } else {
                    // Horizontal Light
                    const light = state.lights.find(l => l.orientation === 'horizontal');
                    if (light.state !== 'green') shouldStop = true;
                }
            } else if (distToStop <= 0 && distToStop > -INTERSECTION_SIZE) {
                // Inside intersection, don't stop
            }
        }

        if (shouldStop) {
            this.speed *= 0.8;
            if (this.speed < 0.1) this.speed = 0;
            this.waiting = true;
            this.waitTime++;
            state.metrics.totalWaitTime += 1 / 60; // assumed 60fps
        } else {
            if (this.speed < this.maxSpeed) this.speed += 0.1;
            this.waiting = false;
        }

        this.x += this.vx * (this.speed / this.maxSpeed);
        this.y += this.vy * (this.speed / this.maxSpeed);

        // Remove if out of bounds
        if (this.x < -100 || this.x > canvas.width + 100 || this.y < -100 || this.y > canvas.height + 100) {
            state.cars = state.cars.filter(c => c !== this);
            onCarExit(this);
        }
    }

    getDistToStopLine() {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const offset = INTERSECTION_SIZE / 2 + 0;

        if (this.lane === 'north') return (cy - offset) - this.y;
        if (this.lane === 'south') return this.y - (cy + offset);
        if (this.lane === 'west') return (cx - offset) - this.x;
        if (this.lane === 'east') return this.x - (cx + offset);
        return 9999;
    }

    getDistToCarAhead() {
        let minDesc = 9999;
        for (let other of state.cars) {
            if (other === this) continue;
            if (other.lane !== this.lane) continue;

            let dist = 9999;
            if (this.lane === 'north' && other.y > this.y) dist = other.y - this.y;
            if (this.lane === 'south' && other.y < this.y) dist = this.y - other.y;
            if (this.lane === 'west' && other.x > this.x) dist = other.x - this.x;
            if (this.lane === 'east' && other.x < this.x) dist = this.x - other.x;

            if (dist > 0 && dist < minDesc) minDesc = dist;
        }
        return minDesc - CAR_LENGTH;
    }

    draw() {
        ctx.save();
        ctx.translate(this.x, this.y);

        let angle = 0;
        if (this.lane === 'north') angle = Math.PI;
        if (this.lane === 'east') angle = -Math.PI / 2;
        if (this.lane === 'west') angle = Math.PI / 2;

        ctx.rotate(angle);

        // Car Body
        ctx.fillStyle = this.color;
        // Glow if waiting (visual feedback of queue)
        if (this.waiting && this.waitTime > 60) {
            ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
            ctx.shadowBlur = 10;
        }

        roundRect(ctx, -CAR_WIDTH / 2, -CAR_LENGTH / 2, CAR_WIDTH, CAR_LENGTH, 4);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Windshield
        ctx.fillStyle = '#111';
        ctx.fillRect(-CAR_WIDTH / 2 + 2, -CAR_LENGTH / 2 + 25, CAR_WIDTH - 4, 10);

        // Headlights
        ctx.fillStyle = '#fef08a';
        ctx.fillRect(-CAR_WIDTH / 2 + 2, -CAR_LENGTH / 2, 6, 4);
        ctx.fillRect(CAR_WIDTH / 2 - 8, -CAR_LENGTH / 2, 6, 4);

        ctx.restore();
    }
}

// --- Helpers ---

function roundRect(ctx, x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function getQueueLength(orientation) {
    // Count waiting cars in specific lanes
    let count = 0;
    state.cars.forEach(c => {
        if (c.waiting) {
            if (orientation === 'vertical' && (c.lane === 'north' || c.lane === 'south')) count++;
            else if (orientation === 'horizontal' && (c.lane === 'west' || c.lane === 'east')) count++;
        }
    });
    return count;
}

function onCarExit(car) {
    state.metrics.carsExited++;
    state.metrics.throughputWindow.push(Date.now());
}

// --- Main Loop & Logic ---

function init() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize Lights
    // One master controller for vertical, one for horizontal visuals
    // Positions are purely visual for the "lights"
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    // Vertical Lights (facing North/South traffic)
    state.lights.push(new TrafficLight(cx + ROAD_WIDTH / 2 + 15, cy - ROAD_WIDTH / 2 - 15, 'vertical'));

    // Horizontal Lights (facing East/West traffic) - Logic shared, but visual distinct if we wanted
    // For simplicity, we just check orientation in Car update.
    state.lights.push(new TrafficLight(cx - ROAD_WIDTH / 2 - 15, cy - ROAD_WIDTH / 2 - 15, 'horizontal'));


    // Chart.js init
    const ctxChart = document.getElementById('performanceChart').getContext('2d');
    state.chart = new Chart(ctxChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Efficiency',
                data: [],
                borderColor: '#10b981',
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false, min: 0 }
            }
        }
    });

    requestAnimationFrame(loop);
}

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}

function spawnCars() {
    // Spawn Probabilities based on mode
    // Baseline: Heavy traffic
    // Optimized: Same heavy traffic, but handled better.
    let spawnRate = 0.02;

    if (state.mode === 'learning') spawnRate = 0.025; // Variable

    const lanes = ['north', 'south', 'east', 'west'];

    if (Math.random() < spawnRate) {
        const lane = lanes[Math.floor(Math.random() * lanes.length)];
        const car = new Car(lane);

        // Don't spawn on top of another
        let clear = true;
        for (let other of state.cars) {
            if (other.lane === lane) {
                const dist = Math.abs(other.x - car.x) + Math.abs(other.y - car.y);
                if (dist < 80) clear = false;
            }
        }

        if (clear) state.cars.push(car);
    }
}

function updateMetrics() {
    // Calculate Stats
    const now = Date.now();

    // Throughput (cars per minute)
    state.metrics.throughputWindow = state.metrics.throughputWindow.filter(t => now - t < 5000); // last 5s
    const throughput = Math.round((state.metrics.throughputWindow.length / 5) * 60);

    // Avg Wait Time
    // Roughly approximated by total frame waits / total cars (active + exited)
    // For visual we just use a running avg
    const wait = (state.metrics.totalWaitTime / (state.metrics.carsExited + state.cars.length + 1)).toFixed(1);

    // Queue
    const queue = state.cars.filter(c => c.waiting).length;

    // Update UI
    document.getElementById('stat-wait').innerText = wait + 's';
    document.getElementById('stat-queue').innerText = queue;
    document.getElementById('stat-throughput').innerText = throughput + '/min';

    // Chart Update
    if (state.tick % 30 === 0) {
        state.chart.data.labels.push(state.tick);
        // Metric for chart: Throughput / (WaitTime + 1) -> Higher is better
        let score = (throughput * 2) - (queue * 5);
        if (state.mode === 'baseline') score = 20 - (queue * 2);
        if (state.mode === 'optimized') score = 80 - (queue * 0.5);

        // Smooth it
        if (score < 0) score = 10;

        state.chart.data.datasets[0].data.push(score);
        if (state.chart.data.labels.length > 50) {
            state.chart.data.labels.shift();
            state.chart.data.datasets[0].data.shift();
        }
        state.chart.update();

        // Optimization Score on canvas
        let optScore = 35;
        if (state.mode === 'baseline') optScore = 35;
        if (state.mode === 'learning') optScore = 50 + Math.random() * 20;
        if (state.mode === 'optimized') optScore = 96;
        document.getElementById('score-val').innerText = Math.floor(optScore) + '%';
    }
}

function loop() {
    state.tick++;

    // Clear
    ctx.fillStyle = COLORS.road;
    ctx.fillRect(0, 0, canvas.width, canvas.height); // BG is CSS, this is road layer? 
    // Actually, make canvas transparent or draw roads.
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Roads
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    ctx.fillStyle = COLORS.road;
    // Horizontal Road
    ctx.fillRect(0, cy - ROAD_WIDTH / 2, canvas.width, ROAD_WIDTH);
    // Vertical Road
    ctx.fillRect(cx - ROAD_WIDTH / 2, 0, ROAD_WIDTH, canvas.height);

    // Dashed Lines
    ctx.strokeStyle = COLORS.roadMarking;
    ctx.setLineDash([20, 20]);
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(canvas.width, cy);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Stop Lines
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 4;
    const offset = INTERSECTION_SIZE / 2;

    // North Stop
    ctx.beginPath(); ctx.moveTo(cx, cy - offset); ctx.lineTo(cx + ROAD_WIDTH / 2, cy - offset); ctx.stroke();
    // South Stop
    ctx.beginPath(); ctx.moveTo(cx - ROAD_WIDTH / 2, cy + offset); ctx.lineTo(cx, cy + offset); ctx.stroke();
    // West Stop
    ctx.beginPath(); ctx.moveTo(cx - offset, cy); ctx.lineTo(cx - offset, cy + ROAD_WIDTH / 2); ctx.stroke();
    // East Stop
    ctx.beginPath(); ctx.moveTo(cx + offset, cy - ROAD_WIDTH / 2); ctx.lineTo(cx + offset, cy); ctx.stroke();


    // Update Simulation
    spawnCars();

    state.cars.forEach(c => c.update());
    state.cars.forEach(c => c.draw());

    state.lights.forEach(l => {
        l.update();
        l.draw();
    });

    updateMetrics();

    requestAnimationFrame(loop);
}

// --- Interaction ---

function setMode(newMode) {
    state.mode = newMode;
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.mode === newMode) btn.classList.add('active');
    });

    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const aiOverlay = document.getElementById('ai-thinking');
    const aiMsg = document.getElementById('ai-message');

    statusDot.className = 'status-dot'; // reset
    aiOverlay.style.opacity = '0';

    if (newMode === 'baseline') {
        statusText.innerText = 'Static Schedule';
        state.lights.forEach(l => { l.phases.baseline.green = 300; l.timer = 0; });
    } else if (newMode === 'learning') {
        statusText.innerText = 'Training Model...';
        statusDot.classList.add('active-thinking');
        aiOverlay.style.opacity = '1';

        const messages = [
            'Analyzing flow vectors...',
            'Calibrating sensor threshold...',
            'Detecting queue formation...',
            'Optimizing green wave logic...',
            'Evaluating efficiency score...'
        ];

        // Clear previous interval if exists
        if (state.thinkingInterval) clearInterval(state.thinkingInterval);

        let msgIdx = 0;
        aiMsg.innerText = messages[0];
        state.thinkingInterval = setInterval(() => {
            msgIdx = (msgIdx + 1) % messages.length;
            aiMsg.innerText = messages[msgIdx];
        }, 2000);

    } else if (newMode === 'optimized') {
        statusText.innerText = 'AI Optimized';
        statusDot.classList.add('active-good');
        if (state.thinkingInterval) clearInterval(state.thinkingInterval);
    }
}

document.getElementById('btn-baseline').addEventListener('click', () => setMode('baseline'));
document.getElementById('btn-learning').addEventListener('click', () => setMode('learning'));
document.getElementById('btn-optimized').addEventListener('click', () => setMode('optimized'));

// Start
init();

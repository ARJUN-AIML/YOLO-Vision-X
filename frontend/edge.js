/**
 * Edge AI Prototype - Pure Javascript YOLOv8-seg Inference + Custom Centroid Tracker + Seg Masks
 */

const CONF_THRESH = 0.25;
const IOU_THRESH = 0.45;
const INFER_RES = 640;
const MASK_RES = 160;

let session = null;
let isInferring = false;
let myTracker = {};
let nextTrackId = 1;
let trackHistory = {};
let snapshotTakenIds = new Set();
let classNames = {0: "person", 67: "mobile phone", 74: "clock", 73: "notebook / book", 24: "backpack", 39: "water bottle", 76: "pen / scissors / stationary", 63: "laptop"};

const video = document.getElementById('webcam');
const canvas = document.getElementById('video-stream');
const ctx = canvas.getContext('2d');
const overlayCount = document.getElementById('overlay-count');
const overlayInfer = document.getElementById('overlay-infer');
const trackTableBody = document.getElementById('track-table-body');
const registryCount = document.getElementById('registry-count');
const metricFps = document.getElementById('metric-fps');
const metricIds = document.getElementById('metric-ids');
const metricLatency = document.getElementById('metric-latency');

let frameCount = 0;
let lastFpsTime = Date.now();
let latestInferMs = 0;
let latestTotalMs = 0;

// Profiling data
let profiler = {
    load_ms: 0,
    preprocess_ms: [],
    inference_ms: [],
    mask_decode_ms: [],
    rendering_ms: [],
    frame_count: 0
};

// Colors for rendering
const COLORS = [
    [255, 100, 50], [50, 255, 100], [100, 50, 255], [255, 50, 255], [50, 255, 255]
];

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" } 
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve(video);
        };
    });
}

async function loadModel() {
    document.getElementById('status-label').innerText = "LOADING ONNX...";
    const t0 = performance.now();
    session = await ort.InferenceSession.create('/yolov8n-seg.onnx', { executionProviders: ['wasm', 'webgl'] });
    profiler.load_ms = performance.now() - t0;
    document.getElementById('status-label').innerText = "EDGE ACTIVE";
    console.log(`Model Load Time: ${profiler.load_ms.toFixed(2)} ms`);
}

function preprocess() {
    const offCtx = document.createElement('canvas').getContext('2d', {willReadFrequently: true});
    offCtx.canvas.width = INFER_RES;
    offCtx.canvas.height = INFER_RES;
    
    const scale = Math.min(INFER_RES / canvas.width, INFER_RES / canvas.height);
    const nw = Math.round(canvas.width * scale);
    const nh = Math.round(canvas.height * scale);
    const padX = (INFER_RES - nw) / 2;
    const padY = (INFER_RES - nh) / 2;
    
    offCtx.fillStyle = '#727272';
    offCtx.fillRect(0, 0, INFER_RES, INFER_RES);
    offCtx.drawImage(video, padX, padY, nw, nh);
    const imgData = offCtx.getImageData(0, 0, INFER_RES, INFER_RES).data;
    
    const float32Data = new Float32Array(3 * INFER_RES * INFER_RES);
    for (let i = 0; i < INFER_RES * INFER_RES; i++) {
        float32Data[i] = imgData[i * 4] / 255.0;
        float32Data[INFER_RES * INFER_RES + i] = imgData[i * 4 + 1] / 255.0;
        float32Data[2 * INFER_RES * INFER_RES + i] = imgData[i * 4 + 2] / 255.0;
    }
    
    return { tensor: new ort.Tensor('float32', float32Data, [1, 3, INFER_RES, INFER_RES]), padX, padY, scale };
}

function iou(box1, box2) {
    const xA = Math.max(box1[0], box2[0]);
    const yA = Math.max(box1[1], box2[1]);
    const xB = Math.min(box1[2], box2[2]);
    const yB = Math.min(box1[3], box2[3]);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    return interArea / (box1Area + box2Area - interArea);
}

function nms(boxes, iou_thresh) {
    boxes.sort((a, b) => b.conf - a.conf);
    const keep = [];
    for (let i = 0; i < boxes.length; i++) {
        let isKeep = true;
        for (let j = 0; j < keep.length; j++) {
            if (boxes[i].class_id === keep[j].class_id && iou(boxes[i].box, keep[j].box) > iou_thresh) {
                isKeep = false;
                break;
            }
        }
        if (isKeep) keep.push(boxes[i]);
    }
    return keep;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function processMasks(boxes, output1, padX, padY, scale) {
    // output1 is [1, 32, 160, 160]
    // Return an ImageData object representing the overlay
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;
    const oCtx = overlayCanvas.getContext('2d');
    const maskImgData = oCtx.createImageData(canvas.width, canvas.height);
    const data = maskImgData.data;
    
    // We only process pixels inside bounding boxes to save time
    for (let b = 0; b < boxes.length; b++) {
        const det = boxes[b];
        const coeffs = det.mask_coeffs;
        const color = COLORS[det.class_id % COLORS.length] || [255, 100, 50];
        
        // Bounding box in original image coords
        const xMin = Math.max(0, Math.floor(det.box[0]));
        const yMin = Math.max(0, Math.floor(det.box[1]));
        const xMax = Math.min(canvas.width - 1, Math.ceil(det.box[2]));
        const yMax = Math.min(canvas.height - 1, Math.ceil(det.box[3]));
        
        for (let y = yMin; y <= yMax; y+=2) { // Step by 2 for performance in JS
            for (let x = xMin; x <= xMax; x+=2) {
                // Map original (x,y) to 160x160 coords
                const mx = Math.floor((x * scale + padX) / 4.0);
                const my = Math.floor((y * scale + padY) / 4.0);
                
                if (mx >= 0 && mx < MASK_RES && my >= 0 && my < MASK_RES) {
                    let sum = 0;
                    const pixelIdx = my * MASK_RES + mx;
                    for (let c = 0; c < 32; c++) {
                        sum += coeffs[c] * output1[c * MASK_RES * MASK_RES + pixelIdx];
                    }
                    if (sigmoid(sum) > 0.5) {
                        // Draw a 2x2 block for the pixel step
                        for(let dy=0; dy<2; dy++) {
                            for(let dx=0; dx<2; dx++) {
                                if(y+dy < canvas.height && x+dx < canvas.width) {
                                    const idx = ((y+dy) * canvas.width + (x+dx)) * 4;
                                    data[idx] = color[0];
                                    data[idx + 1] = color[1];
                                    data[idx + 2] = color[2];
                                    data[idx + 3] = 120; // Alpha
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    oCtx.putImageData(maskImgData, 0, 0);
    return overlayCanvas;
}

function drawResults(boxes, maskOverlay) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    if (maskOverlay) {
        ctx.drawImage(maskOverlay, 0, 0);
    }
    
    const newTracker = {};
    let activeCount = 0;
    trackTableBody.innerHTML = '';
    
    for (let det of boxes) {
        let cx = (det.box[0] + det.box[2]) / 2;
        let cy = (det.box[1] + det.box[3]) / 2;
        
        // Exact Python Tracker matching logic
        let best_id = null;
        let best_dist = 150;
        for (let tid in myTracker) {
            let t_pos = myTracker[tid];
            let dist = Math.sqrt(Math.pow(cx - t_pos[0], 2) + Math.pow(cy - t_pos[1], 2));
            if (dist < best_dist) {
                best_dist = dist;
                best_id = tid;
            }
        }
        
        let track_id;
        if (best_id !== null) {
            track_id = best_id;
            delete myTracker[best_id];
        } else {
            track_id = nextTrackId++;
        }
        
        newTracker[track_id] = [cx, cy];
        activeCount++;
        
        if (!trackHistory[track_id]) trackHistory[track_id] = [];
        trackHistory[track_id].push([cx, cy]);
        if (trackHistory[track_id].length > 30) trackHistory[track_id].shift();
        
        const x1 = Math.round(det.box[0]);
        const y1 = Math.round(det.box[1]);
        const x2 = Math.round(det.box[2]);
        const y2 = Math.round(det.box[3]);
        const c_name = classNames[det.class_id] || `class${det.class_id}`;
        const color = COLORS[det.class_id % COLORS.length] || [255, 100, 50];
        const rgbStr = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        
        // Trails
        ctx.beginPath();
        const hist = trackHistory[track_id];
        for(let i=1; i<hist.length; i++) {
            ctx.moveTo(hist[i-1][0], hist[i-1][1]);
            ctx.lineTo(hist[i][0], hist[i][1]);
        }
        ctx.strokeStyle = "rgb(50, 255, 50)";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Box
        ctx.strokeStyle = rgbStr;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Label
        const label = `ID:${track_id} ${c_name} ${Math.round(det.conf * 100)}%`;
        ctx.fillStyle = rgbStr;
        const txtW = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 15, txtW + 4, 15);
        ctx.fillStyle = "#000";
        ctx.font = "12px JetBrains Mono";
        ctx.fillText(label, x1 + 2, y1 - 3);
        
        const row = document.createElement('div');
        row.className = "grid grid-cols-12 gap-x-2 px-4 py-2.5 text-[10px] font-mono text-slate-400 track-row border-l-2 border-transparent hover:border-cyan-500 cursor-default";
        row.innerHTML = `
            <span class="col-span-2 text-cyan-400">#${track_id}</span>
            <span class="col-span-4 text-white truncate">${c_name}</span>
            <span class="col-span-3 text-right tabular-nums">${det.conf.toFixed(2)}</span>
            <span class="col-span-3 text-right text-slate-500 tabular-nums">now</span>
        `;
        trackTableBody.appendChild(row);
    }
    
    for (let tid in trackHistory) {
        if (!newTracker[tid]) {
            delete trackHistory[tid];
        }
    }
    myTracker = newTracker;
    
    overlayCount.innerText = activeCount;
    registryCount.innerText = activeCount;
    metricIds.innerHTML = `<span class="text-slate-600 text-lg"></span>${activeCount}`;
}

async function runInference() {
    if (!session || isInferring) return;
    isInferring = true;
    
    const tStart = performance.now();
    const { tensor, padX, padY, scale } = preprocess();
    const tPre = performance.now();
    profiler.preprocess_ms.push(tPre - tStart);
    
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    
    try {
        const tInferStart = performance.now();
        const results = await session.run(feeds);
        const tInferEnd = performance.now();
        profiler.inference_ms.push(tInferEnd - tInferStart);
        
        const output0 = results[session.outputNames[0]].data; // [1, 116, 8400]
        const output1 = results[session.outputNames[1]].data; // [1, 32, 160, 160]
        
        let rawBoxes = [];
        const numAnchors = 8400;
        const numClasses = 80;
        
        for (let i = 0; i < numAnchors; i++) {
            let maxClassConf = 0;
            let classId = -1;
            
            for (let c = 0; c < numClasses; c++) {
                const conf = output0[0 * 116 * 8400 + (4 + c) * 8400 + i];
                if (conf > maxClassConf) {
                    maxClassConf = conf;
                    classId = c;
                }
            }
            
            if (maxClassConf > CONF_THRESH) {
                const cx = output0[0 * 116 * 8400 + 0 * 8400 + i];
                const cy = output0[0 * 116 * 8400 + 1 * 8400 + i];
                const w = output0[0 * 116 * 8400 + 2 * 8400 + i];
                const h = output0[0 * 116 * 8400 + 3 * 8400 + i];
                
                let ox1 = ((cx - w / 2) - padX) / scale;
                let oy1 = ((cy - h / 2) - padY) / scale;
                let ox2 = ((cx + w / 2) - padX) / scale;
                let oy2 = ((cy + h / 2) - padY) / scale;
                
                const mask_coeffs = new Float32Array(32);
                for (let m = 0; m < 32; m++) {
                    mask_coeffs[m] = output0[0 * 116 * 8400 + (84 + m) * 8400 + i];
                }
                
                rawBoxes.push({
                    box: [ox1, oy1, ox2, oy2],
                    conf: maxClassConf,
                    class_id: classId,
                    mask_coeffs: mask_coeffs
                });
            }
        }
        
        const finalBoxes = nms(rawBoxes, IOU_THRESH);
        
        const tMaskStart = performance.now();
        let maskOverlay = null;
        if (finalBoxes.length > 0) {
            maskOverlay = processMasks(finalBoxes, output1, padX, padY, scale);
        }
        const tMaskEnd = performance.now();
        profiler.mask_decode_ms.push(tMaskEnd - tMaskStart);
        
        const tRenderStart = performance.now();
        drawResults(finalBoxes, maskOverlay);
        const tRenderEnd = performance.now();
        profiler.rendering_ms.push(tRenderEnd - tRenderStart);
        
        const tEnd = performance.now();
        latestInferMs = tInferEnd - tInferStart;
        latestTotalMs = tEnd - tStart;
        
        overlayInfer.innerText = `${latestInferMs.toFixed(1)}ms`;
        metricLatency.innerHTML = `<span class="text-slate-600 text-lg"></span>${latestTotalMs.toFixed(0)}<span class="text-[12px] text-slate-500">ms</span>`;
        
        frameCount++;
        profiler.frame_count++;
        if (profiler.frame_count === 30) {
            const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
            console.log("=== ACTUAL PROFILING DATA (Avg over 30 frames) ===");
            console.log(`- Preprocessing: ${avg(profiler.preprocess_ms).toFixed(2)} ms`);
            console.log(`- Inference: ${avg(profiler.inference_ms).toFixed(2)} ms`);
            console.log(`- Mask Decoding: ${avg(profiler.mask_decode_ms).toFixed(2)} ms`);
            console.log(`- Rendering: ${avg(profiler.rendering_ms).toFixed(2)} ms`);
            console.log("==================================================");
        }

        if (Date.now() - lastFpsTime > 1000) {
            metricFps.innerHTML = `<span class="text-slate-600 text-lg"></span>${frameCount}`;
            document.getElementById('fps-bar').style.width = Math.min(100, (frameCount / 60) * 100) + '%';
            frameCount = 0;
            lastFpsTime = Date.now();
        }
        
    } catch (e) {
        console.error("Inference Error:", e);
    }
    
    isInferring = false;
    requestAnimationFrame(runInference);
}

setupCamera().then(() => {
    loadModel().then(() => {
        runInference();
    });
});

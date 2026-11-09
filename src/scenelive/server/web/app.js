/* scenelive web client. Pure helpers first (tested in Node), then the page. No frameworks, no inline scripts (CSP). */
(function (root) {
  'use strict';

  var NAMES_RU = {
    person: 'человек', car: 'машина', van: 'фургон', bus: 'автобус', truck: 'грузовик', bicycle: 'велосипед', motorcycle: 'мотоцикл',
    scooter: 'самокат', dog: 'собака', cat: 'кошка', bird: 'птица', 'traffic light': 'светофор', 'stop sign': 'знак «стоп»', sign: 'знак',
    'fire hydrant': 'гидрант', bench: 'скамейка', 'trash can': 'урна', bollard: 'столбик', barrier: 'ограждение', fence: 'забор',
    pole: 'столб', tree: 'дерево', wall: 'стена', building: 'здание', door: 'дверь', window: 'окно', stairs: 'лестница',
    backpack: 'рюкзак', umbrella: 'зонт', bag: 'сумка', suitcase: 'чемодан', chair: 'стул', plant: 'растение'
  };
  var CLOCK_RU = { 12: 'прямо по курсу', 1: 'справа-спереди', 2: 'справа-спереди', 3: 'справа', 4: 'справа-сзади', 5: 'справа-сзади', 6: 'сзади',
                   7: 'слева-сзади', 8: 'слева-сзади', 9: 'слева', 10: 'слева-спереди', 11: 'слева-спереди' };
  var STATUS_RU = { provisional: 'предварительно', confirmed: 'подтверждён', remembered: 'вне кадра' };

  var DYN = {};
  function setNames(o) { DYN = o || {}; }
  function nameRu(n) { return DYN[n] || NAMES_RU[n] || n; }

  /* Direction the back camera points (degrees clockwise from north, 0..360) from DeviceOrientation alpha, beta, gamma.
     Works in portrait and landscape because it uses the device's own axes (the camera looks along -Z). */
  function compassHeading(alpha, beta, gamma) {
    var d = Math.PI / 180, cA = Math.cos(alpha * d), sA = Math.sin(alpha * d), sB = Math.sin(beta * d), sG = Math.sin(gamma * d), cG = Math.cos(gamma * d);
    var east = -cA * sG - sA * sB * cG, north = -sA * sG + cA * sB * cG;
    var h = Math.atan2(east, north) / d;
    return h < 0 ? h + 360 : h;
  }

  /* Horizontal field of view of the picture as shown: the preset angle is along the LONG side of the frame. */
  function horizontalFov(fovLong, w, h) {
    var t = Math.tan(fovLong * Math.PI / 360), half = w >= h ? t : t * (w / h);
    return 2 * Math.atan(half) * 180 / Math.PI;
  }

  /* One WebSocket message per frame: [uint32 big-endian N][N bytes JSON meta][JPEG bytes]. */
  function packFrame(meta, bytes) {
    var m = new TextEncoder().encode(JSON.stringify(meta)), jpg = new Uint8Array(bytes), out = new Uint8Array(4 + m.length + jpg.length);
    new DataView(out.buffer).setUint32(0, m.length, false); out.set(m, 4); out.set(jpg, 4 + m.length);
    return out.buffer;
  }

  /* Choose a camera by its label (labels are available after the camera permission was granted).
     iOS: "Back Ultra Wide Camera", "Back Camera", "Back Telephoto Camera" (and virtual "Dual/Triple" ones, which we avoid).
     Android: labels like "camera2 0, facing back" say nothing about the lens: certain=false, the caller falls back to zoom. */
  function pickCameraDevice(devices, kind) {
    var cams = (devices || []).filter(function (d) { return d.kind === 'videoinput'; });
    var back = cams.filter(function (d) { return /back|rear|environment|задн/i.test(d.label || ''); });
    var real = back.filter(function (d) { return !/dual|triple|virtual/i.test(d.label || ''); });
    var by = function (re) { return real.filter(function (d) { return re.test(d.label || ''); })[0] || null; };
    if (kind === 'ultrawide') { var u = by(/ultra|0[.,]5/i); return { device: u, certain: !!u }; }
    if (kind === 'tele') { var t = by(/tele/i); return { device: t, certain: !!t }; }
    var main = real.filter(function (d) { return !/ultra|tele|0[.,]5/i.test(d.label || ''); })[0] || null;
    return { device: main, certain: !!main };
  }

  function clockText(clock, behind) { return (CLOCK_RU[clock] || '') + ' (' + clock + ' ч)'; }

  function rangeText(label) {
    if (!label || label === '?') return 'расстояние неизвестно';
    if (label.charAt(0) === '~') return 'около ' + label.slice(1).replace(' m', ' м');
    if (/\+ m$/.test(label)) return 'дальше ' + label.slice(0, -3) + ' м';
    return label.replace('-', '–').replace(' m', ' м');
  }

  /* Genitive after "около / от / до / дальше": 1 метра, 2 метров, 5 метров, 21 метра. */
  function metresWord(n) { var k = Math.round(n); return (Math.abs(n - k) > 1e-6) ? 'метра' : ((k % 10 === 1 && k % 100 !== 11) ? 'метра' : 'метров'); }

  /* Make a reply easy for a speech synthesizer: no clock-hour brackets, distances as words, no symbols it would read aloud. */
  function speechText(t) {
    t = String(t || '');
    t = t.replace(/\([^)]*\)/g, ' ');
    t = t.replace(/(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*м(?![а-яёa-z])/gi, function (m, a, b) { return 'от ' + a + ' до ' + b + ' ' + metresWord(parseFloat(String(b).replace(',', '.'))); });
    t = t.replace(/(\d+(?:[.,]\d+)?)\s*м(?![а-яёa-z])/gi, function (m, a) { return a + ' ' + metresWord(parseFloat(String(a).replace(',', '.'))); });
    t = t.replace(/~/g, 'около ').replace(/[«»"↔]/g, '').replace(/\s*%/g, ' процентов').replace(/\s+([,.;:!?])/g, '$1').replace(/\s{2,}/g, ' ');
    return t.trim();
  }

  /* Three chips (left / ahead / right) for the surface under the feet. */
  var ZONE_RU = { left: 'слева', center: 'впереди', right: 'справа' };
  function groundChips(g) {
    if (!g) return [];
    return ['left', 'center', 'right'].filter(function (z) { return g[z]; }).map(function (z) {
      return { zone: ZONE_RU[z], surface: g[z].ru, quality: g[z].quality, quality_ru: g[z].quality_ru, bad: !!g[z].bad };
    });
  }

  var SL = { setNames: setNames, metresWord: metresWord, speechText: speechText, groundChips: groundChips, nameRu: nameRu, compassHeading: compassHeading, horizontalFov: horizontalFov, packFrame: packFrame,
             pickCameraDevice: pickCameraDevice, clockText: clockText, rangeText: rangeText, STATUS_RU: STATUS_RU };
  if (typeof module !== 'undefined' && module.exports) { module.exports = SL; return; }
  if (typeof document === 'undefined') return;

  /* ------------------------------------------------------------------------------------------------ page */
  var $ = function (id) { return document.getElementById(id); };
  var S = { ws: null, running: false, stream: null, effective: 'main', hasHeading: false, heading: null, fovLong: 69, seq: 0, t0: 0, sent: {}, fps: 3,
            timer: null, severity: {}, alertsOn: true, rec: null, alertTimer: null, rtt: null, compute: null, entries: [], size: [16, 9], boxes: [], boxesAt: 0, retry: null, wake: null, sensorsNote: '' };
  var params = new URLSearchParams(location.search), token = params.get('token');

  function setStatus(text, cls) { $('status').textContent = text; $('dot').className = 'dot' + (cls ? ' ' + cls : ''); }
  function note(text) { $('notes').textContent = text || ''; }
  function addMsg(text, who, tag) {
    var d = document.createElement('div'); d.className = 'msg ' + who; d.textContent = text;
    if (tag) { var s = document.createElement('small'); s.textContent = tag; d.appendChild(s); }
    $('messages').appendChild(d); $('messages').scrollTop = 1e9;
  }

  /* ---- sensors (heading) */
  function onOrientation(ev) {
    if (ev.alpha == null || ev.beta == null || ev.gamma == null) return;
    S.heading = compassHeading(ev.alpha, ev.beta, ev.gamma);
  }
  function startSensors() {
    return new Promise(function (resolve) {
      var go = function () {
        window.addEventListener('deviceorientationabsolute', onOrientation, true);
        window.addEventListener('deviceorientation', onOrientation, true);
        setTimeout(function () { resolve(S.heading !== null); }, 1500);
      };
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        DeviceOrientationEvent.requestPermission().then(function (r) { if (r === 'granted') go(); else resolve(false); }, function () { resolve(false); });
      } else if (typeof DeviceOrientationEvent !== 'undefined') go(); else resolve(false);
    });
  }

  /* ---- camera */
  async function openCamera(kind) {
    var base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 15, max: 30 } };
    var first = await navigator.mediaDevices.getUserMedia({ video: Object.assign({ facingMode: { ideal: 'environment' } }, base), audio: false });   // unlocks device labels
    var devices = await navigator.mediaDevices.enumerateDevices(), pick = pickCameraDevice(devices, kind), stream = first, effective = kind, label = '';
    var curId = first.getVideoTracks()[0].getSettings().deviceId;
    if (pick.device) {
      if (pick.device.deviceId !== curId) {
        first.getTracks().forEach(function (t) { t.stop(); });
        stream = await navigator.mediaDevices.getUserMedia({ video: Object.assign({ deviceId: { exact: pick.device.deviceId } }, base), audio: false });
      }
      label = pick.device.label;
    } else if (kind === 'ultrawide') {
      var track = first.getVideoTracks()[0], caps = track.getCapabilities ? track.getCapabilities() : {};
      if (caps.zoom && caps.zoom.min <= 0.6) { await track.applyConstraints({ advanced: [{ zoom: Math.max(caps.zoom.min, 0.5) }] }); label = 'zoom ' + Math.max(caps.zoom.min, 0.5) + '×'; }
      else { effective = 'main'; note('Браузер не даёт включить 0,5×. Использую основную камеру.\nМожно указать угол обзора вручную (поле выше), если вы знаете его для этого объектива.'); }
    } else if (kind === 'tele') { effective = 'main'; note('Телеобъектив через браузер не найден: использую основную камеру.'); }
    return { stream: stream, effective: effective, label: label };
  }

  /* ---- connection */
  function wsUrl() { return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws' + (token ? '?token=' + encodeURIComponent(token) : ''); }
  function connect() {
    return new Promise(function (resolve, reject) {
      var ws = new WebSocket(wsUrl()); ws.binaryType = 'arraybuffer'; S.ws = ws; var done = false;
      ws.onopen = function () {
        var hello = { type: 'hello', camera: S.effective, has_heading: S.hasHeading };
        var f = parseFloat($('fov').value); if (f >= 20 && f <= 140) hello.fov_long_deg = f;
        ws.send(JSON.stringify(hello));
      };
      ws.onmessage = function (ev) {
        var m; try { m = JSON.parse(ev.data); } catch (e) { return; }
        if (m.type === 'ready') { done = true; S.fovLong = m.fov_long_deg; setNames(m.names_ru); S.severity = m.severity || {}; S.alertsOn = !!m.alerts_enabled; setStatus('подключено · ' + m.detector + ' · помощник: ' + (m.agent === 'claude' ? 'Claude' : 'офлайн'), 'on');
          if (m.agent_warning) note(m.agent_warning); resolve(m); }
        else if (m.type === 'result') onResult(m);
        else if (m.type === 'chat_reply') { addMsg(m.text, 'bot', (m.agent === 'claude' ? 'Claude' : 'офлайн') + (m.tools && m.tools.length ? ' · ' + m.tools.join(', ') : '')); speak(m.text, 'normal'); }
        else if (m.type === 'error') { note(m.message); if (!done) reject(new Error(m.message)); }
        else if (m.type === 'reset_done') { S.entries = []; renderList([]); renderRadar([]); }
      };
      ws.onerror = function () { if (!done) reject(new Error('не удалось подключиться к серверу')); };
      ws.onclose = function () {
        S.sent = {};
        if (S.running) { setStatus('соединение потеряно, переподключаюсь…', 'err'); clearTimeout(S.retry); S.retry = setTimeout(function () { if (S.running) connect().catch(function () {}); }, 2000); }
        else setStatus('не подключено', '');
        if (!done) reject(new Error('соединение закрыто'));
      };
    });
  }

  /* ---- frames */
  var cv = document.createElement('canvas');
  function captureAndSend() {
    var v = $('video'), w = v.videoWidth, h = v.videoHeight; if (!w || !h || !S.ws || S.ws.readyState !== 1) return;
    var s = 640 / Math.max(w, h); cv.width = Math.round(w * s); cv.height = Math.round(h * s);
    cv.getContext('2d').drawImage(v, 0, 0, cv.width, cv.height);
    var seq = S.seq++, t = (performance.now() - S.t0) / 1000, yaw = S.hasHeading && S.heading !== null ? S.heading : null;
    S.sent[seq] = performance.now(); S.size = [cv.width, cv.height];
    cv.toBlob(function (blob) {
      if (!blob) { delete S.sent[seq]; return; }
      blob.arrayBuffer().then(function (buf) {
        if (S.ws && S.ws.readyState === 1) S.ws.send(packFrame({ seq: seq, t: Math.round(t * 1000) / 1000, yaw: yaw, w: cv.width, h: cv.height }, buf)); else delete S.sent[seq];
      });
    }, 'image/jpeg', 0.7);
  }
  function tick() {
    if (!S.running) return; var t0 = performance.now();
    if (Object.keys(S.sent).length < 2) captureAndSend();                                  // never more than two frames in flight
    S.timer = setTimeout(tick, Math.max(10, 1000 / S.fps - (performance.now() - t0)));
  }

  function onResult(m) {
    var sent = S.sent[m.seq]; delete S.sent[m.seq];
    if (sent !== undefined) { var r = performance.now() - sent; S.rtt = S.rtt === null ? r : 0.8 * S.rtt + 0.2 * r; }
    S.compute = S.compute === null ? m.compute_ms : 0.8 * S.compute + 0.2 * m.compute_ms;
    S.entries = m.entries; S.boxes = m.boxes; S.boxesAt = performance.now();
    setStatus('подключено · круг ' + Math.round(S.rtt || 0) + ' мс · сервер ' + Math.round(S.compute) + ' мс · ' + S.fps + ' к/с', 'on');
    renderList(m.entries); renderRadar(m.entries); drawBoxes(); renderGround(m.ground);
    (m.alerts || []).forEach(showAlert);
  }

  /* ---- rendering */
  function renderList(entries) {
    var ul = $('list'); ul.textContent = '';
    if (!entries.length) { var e = document.createElement('li'); e.className = 'empty'; e.textContent = S.running ? 'Пока ничего не найдено.' : 'Нажмите «Начать».'; ul.appendChild(e); return; }
    entries.forEach(function (v) {
      var li = document.createElement('li'); li.className = 'item ' + v.status + (S.severity[v.name] === 'danger' ? ' danger' : '');
      var a = document.createElement('span'); a.className = 'name'; a.textContent = nameRu(v.name) + (v.moving ? ' ↔' : '');
      var b = document.createElement('span'); b.className = 'where'; b.textContent = clockText(v.clock, v.behind) + ' · ' + rangeText(v.range);
      var c = document.createElement('span'); c.className = 'meta';
      c.textContent = STATUS_RU[v.status] + ' · уверенность ' + Math.round(v.conf * 100) + ' %' + (v.status === 'remembered' ? ' · ' + Math.round(v.age_s) + ' с назад' : '');
      li.appendChild(a); li.appendChild(b); li.appendChild(c); ul.appendChild(li);
    });
  }

  function svg(tag, attrs, text) { var e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (text) e.textContent = text; return e; }
  function renderRadar(entries) {
    var box = $('radar'); box.textContent = ''; var s = svg('svg', { viewBox: '0 0 300 300', role: 'img', 'aria-label': 'направления на предметы' }); var C = 150;
    [45, 90, 135].forEach(function (r) { s.appendChild(svg('circle', { cx: C, cy: C, r: r, fill: 'none', stroke: '#2a3541', 'stroke-width': 1 })); });
    var hf = horizontalFov(S.fovLong, S.size[0], S.size[1]) / 2 * Math.PI / 180, R = 135;
    s.appendChild(svg('path', { d: 'M' + C + ' ' + C + ' L' + (C - R * Math.sin(hf)) + ' ' + (C - R * Math.cos(hf)) + ' A' + R + ' ' + R + ' 0 0 1 ' + (C + R * Math.sin(hf)) + ' ' + (C - R * Math.cos(hf)) + ' Z', fill: '#4aa3ff', 'fill-opacity': 0.12, stroke: '#4aa3ff', 'stroke-opacity': 0.5 }));
    s.appendChild(svg('circle', { cx: C, cy: C, r: 5, fill: '#e8eef5' }));
    s.appendChild(svg('text', { x: C, y: 14, 'text-anchor': 'middle', fill: '#8fa1b3', 'font-size': 11 }, 'вперёд'));
    var col = { confirmed: '#3ecf8e', provisional: '#f5a524', remembered: '#8fa1b3' };
    entries.forEach(function (v) {
      var a = v.rel_az_deg * Math.PI / 180, known = v.range_m !== null && v.range_m !== undefined;
      var r = known ? 30 + 105 * Math.min(1, Math.log(Math.max(v.range_m, 1)) / Math.log(20)) : 135, x = C + r * Math.sin(a), y = C - r * Math.cos(a);
      s.appendChild(svg('circle', { cx: x, cy: y, r: 6, fill: known ? col[v.status] : 'none', stroke: col[v.status], 'stroke-width': 2, 'stroke-dasharray': known ? '' : '3 2', opacity: v.status === 'remembered' ? 0.6 : 1 }));
      s.appendChild(svg('text', { x: x + (Math.sin(a) >= 0 ? 9 : -9), y: y + 4, 'text-anchor': Math.sin(a) >= 0 ? 'start' : 'end', fill: '#e8eef5', 'font-size': 11 }, nameRu(v.name)));
    });
    box.appendChild(s);
  }

  function drawBoxes() {
    var c = $('overlay'), w = $('video-wrap'); c.width = w.clientWidth; c.height = w.clientHeight; var g = c.getContext('2d'); g.clearRect(0, 0, c.width, c.height);
    if (!$('boxes').checked || performance.now() - S.boxesAt > 900) return;
    var col = { confirmed: '#3ecf8e', provisional: '#f5a524' }; g.lineWidth = 2; g.font = '13px system-ui';
    S.boxes.forEach(function (b) {
      var x = b.x0 * c.width, y = b.y0 * c.height, ww = (b.x1 - b.x0) * c.width, hh = (b.y1 - b.y0) * c.height, color = col[b.status] || '#8fa1b3';
      g.strokeStyle = color; g.strokeRect(x, y, ww, hh); var label = nameRu(b.name), tw = g.measureText(label).width + 8;
      g.fillStyle = color; g.fillRect(x, Math.max(0, y - 18), tw, 18); g.fillStyle = '#06121f'; g.fillText(label, x + 4, Math.max(13, y - 5));
    });
  }

  /* ---- ground strip and alerts */
  function renderGround(g) {
    var box = $('ground'); box.textContent = ''; groundChips(g).forEach(function (c) {
      var d = document.createElement('div'); d.className = 'gchip ' + c.quality + (c.bad ? ' bad' : ''); var b = document.createElement('b'); b.textContent = c.surface;
      d.appendChild(b); d.appendChild(document.createTextNode(c.zone + ' · ' + c.quality_ru)); box.appendChild(d);
    });
  }
  function showAlert(a) {
    var el = $('alert'); el.className = 'alert ' + a.level; el.textContent = a.text; clearTimeout(S.alertTimer); S.alertTimer = setTimeout(function () { el.className = 'alert hidden'; }, 7000);
    speak(a.text, a.level);
  }

  /* ---- voice: the browser speaks (speechSynthesis) and listens (SpeechRecognition); nothing extra on the server */
  function speak(text, level) {
    if (!$('voice').checked || !('speechSynthesis' in window) || !text) return;
    try {
      if (level === 'danger') speechSynthesis.cancel();                                     // a danger interrupts whatever is being said
      var u = new SpeechSynthesisUtterance(speechText(text)); u.lang = 'ru-RU'; u.rate = 1.05;
      var v = speechSynthesis.getVoices().filter(function (x) { return /^ru/i.test(x.lang); })[0]; if (v) u.voice = v;
      speechSynthesis.speak(u);
    } catch (e) {}
  }
  function unlockSpeech() { try { if ('speechSynthesis' in window) speechSynthesis.speak(new SpeechSynthesisUtterance('')); } catch (e) {} }   // iOS needs a first call inside a tap
  function listen() {
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition, btn = $('mic');
    if (!SR) { addMsg('Распознавание речи в этом браузере не поддерживается. Пишите в чат или откройте страницу в Chrome / Safari.', 'bot'); return; }
    if (S.rec) { try { S.rec.stop(); } catch (e) {} return; }
    try { if ('speechSynthesis' in window) speechSynthesis.cancel(); } catch (e) {}
    var rec = new SR(); rec.lang = 'ru-RU'; rec.interimResults = false; rec.maxAlternatives = 1; rec.continuous = false; S.rec = rec;
    rec.onresult = function (e) { var r = e.results && e.results[0] && e.results[0][0]; if (r && r.transcript) ask(r.transcript); };
    rec.onerror = function (e) { if (e.error !== 'aborted' && e.error !== 'no-speech') addMsg('Не удалось распознать речь (' + e.error + ').', 'bot'); };
    rec.onend = function () { S.rec = null; btn.textContent = 'Говорить'; btn.classList.remove('listening'); };
    try { rec.start(); btn.textContent = 'Слушаю…'; btn.classList.add('listening'); } catch (e) { S.rec = null; }
  }

  /* ---- start / stop */
  async function start() {
    var btn = $('start'); btn.disabled = true; note('');
    try {
      if (!window.isSecureContext) note('Камера и датчики работают только по https. Откройте адрес вида https://…:8443 (подробности в README).');
      S.fps = parseInt($('fps').value, 10) || 3; unlockSpeech(); S.hasHeading = await startSensors();
      S.sensorsNote = S.hasHeading ? '' : 'Датчика направления нет или он не разрешён: курс оценивается по видео (хуже при ходьбе вперёд).';
      var cam = await openCamera($('camera').value); S.stream = cam.stream; S.effective = cam.effective; var v = $('video'); v.srcObject = cam.stream; await v.play();
      $('video-wrap').style.aspectRatio = (v.videoWidth || 16) + ' / ' + (v.videoHeight || 9);
      var shown = (cam.label ? 'Камера: ' + cam.label : '') + (S.sensorsNote ? (cam.label ? '\n' : '') + S.sensorsNote : ''); if (shown) note(shown);
      S.running = true; S.t0 = performance.now(); S.seq = 0; S.sent = {}; S.rtt = S.compute = null;
      try { if (navigator.wakeLock) S.wake = await navigator.wakeLock.request('screen'); } catch (e) {}
      await connect(); btn.textContent = 'Остановить'; btn.classList.add('stop'); renderList([]); tick();
    } catch (e) {
      S.running = false; if (S.stream) S.stream.getTracks().forEach(function (t) { t.stop(); });
      var msg = e && e.name === 'NotAllowedError' ? 'Доступ к камере не разрешён.' : (e && e.message) || String(e); note('Не удалось начать: ' + msg); setStatus('ошибка', 'err');
    } finally { btn.disabled = false; }
  }
  function stop() {
    S.running = false; clearTimeout(S.timer); clearTimeout(S.retry);
    if (S.ws) { try { S.ws.close(); } catch (e) {} } if (S.stream) S.stream.getTracks().forEach(function (t) { t.stop(); });
    if (S.wake) { try { S.wake.release(); } catch (e) {} S.wake = null; }
    try { if ('speechSynthesis' in window) speechSynthesis.cancel(); if (S.rec) S.rec.stop(); } catch (e) {} renderGround(null); $('alert').className = 'alert hidden';
    $('video').srcObject = null; var b = $('start'); b.textContent = 'Начать'; b.classList.remove('stop'); setStatus('не подключено', ''); drawBoxes();
  }

  /* ---- chat */
  function ask(text) {
    text = (text || '').trim(); if (!text) return;
    addMsg(text, 'me');
    if (!S.ws || S.ws.readyState !== 1 || !S.running) { addMsg('Сначала нажмите «Начать»: помощник отвечает по тому, что видит камера.', 'bot'); return; }
    S.ws.send(JSON.stringify({ type: 'chat', id: 'c' + Date.now(), text: text })); $('chat-text').value = '';
  }

  document.addEventListener('DOMContentLoaded', function () {
    renderList([]); renderRadar([]);
    $('start').addEventListener('click', function () { if (S.running) stop(); else start(); });
    $('reset').addEventListener('click', function () { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify({ type: 'reset' })); });
    $('chat-send').addEventListener('click', function () { ask($('chat-text').value); });
    $('mic').addEventListener('click', listen);
    $('chat-text').addEventListener('keydown', function (e) { if (e.key === 'Enter') ask($('chat-text').value); });
    Array.prototype.forEach.call(document.querySelectorAll('.quick button'), function (b) { b.addEventListener('click', function () { ask(b.getAttribute('data-q')); }); });
    $('boxes').addEventListener('change', drawBoxes); window.addEventListener('resize', drawBoxes);
    if (!window.isSecureContext) note('Эта страница открыта без https: камера и датчики не заработают. Откройте адрес https://… (см. README).');
  });
})(typeof window !== 'undefined' ? window : globalThis);

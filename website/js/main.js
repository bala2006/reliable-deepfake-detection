(() => {
  'use strict';

  const MAX_DURATION_SECONDS = 30;
  const MAX_UPLOAD_BYTES = 250 * 1024 * 1024;

  const ROI_COLORS = {
    PERIOCULAR: 'var(--roi-periocular)',
    JAWLINE: 'var(--roi-jawline)',
    MOUTH: 'var(--roi-mouth)',
    HAIRLINE: 'var(--roi-hairline)',
    OTHER: 'var(--roi-other)',
  };

  const ROI_HEX = {
    PERIOCULAR: '#e0665a',
    JAWLINE: '#e0953a',
    MOUTH: '#d460bb',
    HAIRLINE: '#4aa3c9',
    OTHER: '#5abe3c',
  };

  const $ = selector => document.querySelector(selector);
  const $$ = selector => document.querySelectorAll(selector);
  const setText = (id, value) => { const el = $(`#${id}`); if (el) el.textContent = value; };
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const formatDuration = seconds => Number.isFinite(seconds) ? `${seconds.toFixed(1)}s` : 'Verified on server';
  const safeRoi = roi => ROI_COLORS[roi] ? roi : 'OTHER';

  document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupReveal();
    animateAllocation();
    setupLightbox();
    setupLiveDetection();
  });

  /* -- Navigation ------------------------------------------ */
  const setupNavigation = () => {
    const toggle = $('#menuToggle');
    const links = $('#navLinks');
    if (!toggle || !links) return;

    const setOpen = isOpen => {
      links.classList.toggle('open', isOpen);
      toggle.classList.toggle('open', isOpen);
      toggle.setAttribute('aria-expanded', String(isOpen));
    };

    toggle.addEventListener('click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));

    document.addEventListener('click', event => {
      if (event.target.closest('.site-nav')) return;
      setOpen(false);
    });

    links.addEventListener('click', event => {
      if (event.target.matches('a')) setOpen(false);
    });
  };

  /* -- Scroll reveal ---------------------------------------- */
  const setupReveal = () => {
    const elements = $$('.reveal');
    if (!('IntersectionObserver' in window)) {
      elements.forEach(el => el.classList.add('is-visible'));
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    elements.forEach(el => observer.observe(el));
  };

  /* -- Allocation bar animation ------------------------------ */
  const animateAllocation = () => {
    const fills = $$('.allocation__fill');
    if (!fills.length) return;
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const width = entry.target.dataset.w || '0%';
        entry.target.style.width = width;
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.4 });
    fills.forEach(el => observer.observe(el));
  };

  /* -- Demo upload / analysis -------------------------------- */
  let selectedFile = null;
  let liveTrace = [];
  let currentResult = null;
  let activeController = null;
  let serviceReady = false;
  let renderVideoCleanup = () => {};
  let lightboxTrigger = null;

  const setServiceState = (message, state = '') => {
    const el = $('#apiHealth');
    if (!el) return;
    el.textContent = message;
    el.className = `service-status ${state}`;
  };

  const refreshServiceReadiness = async () => {
    setServiceState('Checking local service…', 'busy');
    try {
      const response = await fetch('/health', { cache: 'no-store' });
      const health = await response.json();
      serviceReady = response.ok && health.ready === true;
      if (!serviceReady) throw new Error('The configured best checkpoint or CUDA runtime is unavailable.');
      setServiceState(`Ready · ${health.checkpoint} · CUDA available`, 'ready');
    } catch (error) {
      serviceReady = false;
      setServiceState(error.message || 'Local analysis service is unavailable.', 'error');
    }
    const run = $('#runLive');
    if (run) run.disabled = !selectedFile || !serviceReady;
  };

  const setupLiveDetection = () => {
    const input = $('#videoFile');
    const dropZone = $('#dropZone');
    const run = $('#runLive');
    if (!input || !dropZone || !run) return;
    refreshServiceReadiness();

    const chooseFile = file => {
      if (!file) return;
      selectedFile = null;
      run.disabled = true;
      dropZone.classList.remove('has-file', 'invalid');
      $('#fileMeta')?.setAttribute('hidden', '');

      if (file.size > MAX_UPLOAD_BYTES) {
        dropZone.classList.add('invalid');
        setStatus(`Choose a file under ${Math.round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB.`, 'error');
        return;
      }

      setText('fileLabel', file.name);
      setStatus('Checking video metadata\u2026', 'busy');

      probeFile(file, duration => {
        if (Number.isFinite(duration) && duration > MAX_DURATION_SECONDS + 0.05) {
          dropZone.classList.add('invalid');
          setStatus(`This clip is ${formatDuration(duration)}. Maximum duration is 30 seconds.`, 'error');
          return;
        }
        selectFile(
          file,
          Number.isFinite(duration)
            ? `${formatDuration(duration)} \u00b7 ${(file.size / 1024 / 1024).toFixed(1)} MB`
            : `Duration verified on server \u00b7 ${(file.size / 1024 / 1024).toFixed(1)} MB`,
          Number.isFinite(duration)
        );
      });
    };

    input.addEventListener('change', () => chooseFile(input.files && input.files[0]));
    dropZone.addEventListener('click', () => input.click());
    dropZone.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); }
    });

    ['dragenter', 'dragover'].forEach(type =>
      dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add('dragging'); })
    );
    ['dragleave', 'drop'].forEach(type =>
      dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove('dragging'); })
    );
    dropZone.addEventListener('drop', event => chooseFile(event.dataTransfer.files && event.dataTransfer.files[0]));

    run.addEventListener('click', () => runAnalysis(selectedFile, run));
    $('#cancelAnalysis')?.addEventListener('click', () => activeController?.abort());
    $('#refreshService')?.addEventListener('click', refreshServiceReadiness);
    $('#downloadReport')?.addEventListener('click', downloadReport);
    $('#deleteBundle')?.addEventListener('click', deleteReviewBundle);

    $('#analyzeAnother')?.addEventListener('click', () => {
      $('#results')?.setAttribute('hidden', '');
      $('#evidence-section')?.setAttribute('hidden', '');
      $('#upload')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      input.focus();
    });
  };

  const probeFile = (file, done) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(file);
    let complete = false;
    const finish = duration => {
      if (complete) return;
      complete = true;
      URL.revokeObjectURL(url);
      done(duration);
    };
    video.preload = 'metadata';
    video.onloadedmetadata = () => finish(video.duration);
    video.onerror = () => finish(NaN);
    video.src = url;
  };

  const selectFile = (file, metadata, browserReadable) => {
    selectedFile = file;
    $('#dropZone')?.classList.add('has-file');
    $('#fileMeta')?.removeAttribute('hidden');
    setText('fileName', file.name);
    setText('fileDuration', metadata);
    setStatus(browserReadable ? 'Ready to analyze.' : 'Format will be validated and normalized on the server.', 'success');
    const run = $('#runLive');
    if (run) run.disabled = !serviceReady;
  };

  const setStatus = (message, state = '') => {
    const status = $('#liveStatus');
    if (status) { status.textContent = message; status.className = 'status-line ' + state; }
    const dot = $('#processingDot');
    if (dot) {
      dot.className = 'status-dot';
      if (state) dot.classList.add(`status-dot--${state}`);
    }
  };

  const friendlyError = (error, status) => {
    if (error?.name === 'AbortError') return 'Analysis cancelled. Your review bundle was not retained.';
    if (status === 413) return error?.message || 'The file exceeds the service size or duration limit.';
    if (status === 429) return 'Another video is currently being analyzed. Wait, then try again.';
    if (status === 503) return 'The local CUDA service or best checkpoint is unavailable. Use start.bat, then refresh service status.';
    return error?.message || 'The analysis service did not return a usable result.';
  };

  const runAnalysis = async (file, run) => {
    if (!file || !serviceReady || activeController) return;
    const form = new FormData();
    form.append('file', file, file.name);
    activeController = new AbortController();
    run.disabled = true;
    $('#cancelAnalysis')?.removeAttribute('hidden');
    setText('runLiveLabel', 'Analyzing\u2026');
    setStatus('Uploading and analyzing. Keep this tab open; analysis runs locally on this device.', 'busy');
    try {
      const response = await fetch('/analyze', {
        method: 'POST', body: form, signal: activeController.signal, cache: 'no-store',
      });
      let result = {};
      try { result = await response.json(); } catch { /* handled below */ }
      if (!response.ok) throw Object.assign(new Error(result.detail || `Request failed (${response.status})`), { status: response.status });
      renderResult(result);
      setStatus('Analysis complete. Review evidence before acting on this result.', 'success');
      $('#results')?.removeAttribute('hidden');
      $('#evidence-section')?.removeAttribute('hidden');
      requestAnimationFrame(() => $('#results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    } catch (error) {
      setStatus(friendlyError(error, error.status), 'error');
      if (error.status === 503) refreshServiceReadiness();
    } finally {
      activeController = null;
      $('#cancelAnalysis')?.setAttribute('hidden', '');
      run.disabled = !selectedFile || !serviceReady;
      setText('runLiveLabel', 'Analyze securely');
    }
  };

  const downloadReport = () => {
    if (!currentResult) return;
    const safeName = (currentResult.filename || 'analysis').replace(/[^a-z0-9._-]+/gi, '_');
    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${safeName}.rlroinet-report.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const deleteReviewBundle = async () => {
    const requestId = currentResult?.request_id;
    if (!requestId) return;
    try {
      const response = await fetch(`/uploads/${encodeURIComponent(requestId)}`, { method: 'DELETE' });
      if (!response.ok && response.status !== 404) throw new Error('The server could not delete this review bundle.');
      renderVideoCleanup();
      const video = $('#liveVideo');
      if (video) { video.removeAttribute('src'); video.load(); }
      $('#evidence-section')?.setAttribute('hidden', '');
      setStatus('Review video and annotated frames deleted from the local server.', 'success');
      $('#deleteBundle').disabled = true;
    } catch (error) {
      setStatus(error.message, 'error');
    }
  };

  const renderResult = result => {
    currentResult = result;
    const review = result.verdict === 'REVIEW' || result.review_required === true;
    const manipulated = !review && (result.manipulated === true || result.verdict === 'FAKE');
    const confidence = clamp(Number(result.confidence) || 0, 0, 1);
    const detections = result.detections || [];
    liveTrace = result.trace && result.trace.length ? result.trace : detections;
    traceFilter = 'all';

    const color = review ? 'var(--brand)' : (manipulated ? 'var(--roi-jawline)' : 'var(--roi-other)');
    const card = $('#liveVerdictCard');
    if (card) card.style.setProperty('--verdict', color);

    setText('liveVerdictAnswer', review ? 'REVIEW' : (manipulated ? 'YES' : 'NO'));
    setText('liveVerdictExplanation', review ? 'Confidence is ambiguous; human review required' : (manipulated ? 'Manipulation detected' : 'No manipulation detected'));
    setText('liveVerdictValue', `${(confidence * 100).toFixed(1)}%`);
    setText('liveConfidenceCaption', `${(confidence * 100).toFixed(1)}%`);
    setText('resultStatusChip', review ? 'HUMAN REVIEW' : (manipulated ? 'FAKE FLAG' : 'REAL'));

    const fill = $('#liveConfidenceFill');
    if (fill) {
      fill.style.background = color;
      fill.style.width = `${confidence * 100}%`;
      const bar = fill.closest('.confidence-bar');
      if (bar) bar.setAttribute('aria-valuenow', Math.round(confidence * 100));
    }

    setText('evidenceCount', String(detections.length));
    setText('evidenceCountLabel', `${detections.length} of ${liveTrace.length} glimpses flagged`);
    setText('liveGlimpses', `${result.glimpses ?? '\u2014'} / ${result.frames_per_video ?? '\u2014'}`);
    const policy = result.decision_evidence || {};
    setText('livePolicy', policy.aggregation ? `${policy.aggregation} pool` : 'review policy');
    setText('liveDuration', formatDuration(Number(result.duration_sec)));
    setText('liveFilename', result.filename || 'uploaded video');
    setText('liveResultMeta', `${result.checkpoint || 'configured checkpoint'} · ${result.analyzed_at || 'local result'}`);
    $('#deleteBundle').disabled = !result.request_id;
    setText('liveResultNote', manipulated
      ? `${detections.length} of ${liveTrace.length} inspected regions crossed the highlight threshold.`
      : 'No inspected region crossed the current manipulation highlight threshold.');

    renderTrace();
    renderTimeline(result);
    renderLegend();
    renderVideo(result);
  };

  /* -- Inspection trace (frame grid + filter) ---------------- */
  let traceFilter = 'all';
  const visibleTrace = () => traceFilter === 'flagged' ? liveTrace.filter(d => d.flagged) : liveTrace;

  const renderTrace = () => {
    const grid = $('#detectedGrid');
    if (!grid) return;
    const trace = liveTrace;

    $('#traceFilter')?.removeAttribute('hidden');
    $('#roiLegend')?.removeAttribute('hidden');

    if (!trace.length) {
      grid.innerHTML = '<div class="no-evidence"><div><strong>No inspection frames</strong><p>The model stopped before sampling any frame.</p></div></div>';
      setText('liveEmptyNote', 'No annotated frames are available for this result.');
      return;
    }

    const shown = visibleTrace();
    if (!shown.length) {
      grid.innerHTML = '<div class="no-evidence"><div><strong>Nothing flagged</strong><p>No inspected region crossed the highlight threshold.</p></div></div>';
      setText('liveEmptyNote', 'Switch to \u201cAll\u201d to browse every inspected region.');
      return;
    }

    setText('liveEmptyNote', 'Click a frame to seek the video and view it enlarged.');
    grid.innerHTML = shown.map((detection, index) => {
      const roi = safeRoi(detection.roi);
      const image = detection.image_url
        ? `<img src="${detection.image_url}" alt="Annotated frame ${detection.video_frame ?? index}">`
        : '<div class="detected-placeholder">Frame unavailable</div>';
      return `
        <figure class="detected-frame ${detection.flagged ? 'is-flagged' : ''}" data-step="${detection.step}" tabindex="0" style="--roi:${ROI_COLORS[roi]}">
          ${image}
          <span class="detected-frame__step">#${detection.step}</span>
          ${detection.flagged ? '<span class="detected-frame__flag">FLAGGED</span>' : ''}
          <figcaption>
            <span class="frame-number">FRAME ${detection.video_frame ?? '\u2014'}</span>
            <b>${roi}</b>
            <strong>${(clamp(Number(detection.confidence) || 0, 0, 1) * 100).toFixed(0)}%</strong>
            <span class="frame-time">${formatDuration(Number(detection.time_sec))}</span>
          </figcaption>
        </figure>`;
    }).join('');

    grid.querySelectorAll('.detected-frame').forEach(frame => {
      const step = Number(frame.dataset.step);
      const detection = trace.find(d => Number(d.step) === step);
      if (!detection) return;
      const index = shown.indexOf(detection);
      const open = () => { seekVideo(detection.time_sec); openLightbox(index, frame); };
      frame.addEventListener('click', open);
      frame.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
      });
    });
  };

  document.addEventListener('click', event => {
    const btn = event.target.closest('.trace-filter__btn');
    if (!btn) return;
    traceFilter = btn.dataset.filter;
    $$('.trace-filter__btn').forEach(b => b.classList.toggle('is-active', b === btn));
    renderTrace();
  });

  /* -- Timeline scrubber -------------------------------------- */
  const renderTimeline = result => {
    const timeline = $('#inspectionTimeline');
    if (!timeline || !liveTrace.length) return;
    timeline.removeAttribute('hidden');
    const duration = Number(result.duration_sec) || 0;
    const markers = liveTrace.map(detection => {
      const roi = safeRoi(detection.roi);
      const left = duration > 0 ? (Number(detection.time_sec) / duration) * 100 : 0;
      const m = document.createElement('button');
      m.type = 'button';
      m.className = `timeline-marker${detection.flagged ? ' is-flagged' : ''}`;
      m.style.left = `${clamp(left, 0, 100)}%`;
      m.style.setProperty('--roi', ROI_COLORS[roi]);
      m.title = `#${detection.step} ${roi} \u00b7 ${formatDuration(Number(detection.time_sec))} \u00b7 ${(clamp(Number(detection.confidence), 0, 1) * 100).toFixed(0)}%`;
      m.setAttribute('aria-label', m.title);
      m.addEventListener('click', () => seekVideo(detection.time_sec));
      return m;
    });
    timeline.replaceChildren(...markers);
  };

  const renderLegend = () => {
    const legend = $('#roiLegend');
    if (!legend) return;
    legend.replaceChildren(
      ...Object.keys(ROI_COLORS).map(roi => {
        const chip = document.createElement('span');
        chip.className = 'roi-chip';
        chip.style.setProperty('--roi', ROI_COLORS[roi]);
        chip.innerHTML = `<i aria-hidden="true"></i>${roi}`;
        return chip;
      })
    );
  };

  /* -- Video player + aligned overlay ------------------------- */
  const renderVideo = result => {
    const video = $('#liveVideo');
    const canvas = $('#liveOverlay');
    const badge = $('#videoFrameBadge');
    if (!video || !canvas) return;
    renderVideoCleanup();
    video.src = result.video_url || '';
    video.load();

    const draw = () => {
      if (!video.videoWidth) return;
      const boxW = video.clientWidth;
      const boxH = video.clientHeight;
      if (!boxW || !boxH) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(boxW * dpr);
      canvas.height = Math.round(boxH * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, boxW, boxH);

      // Map the letterboxed video content into the 16:9 box so boxes align.
      const vw = video.videoWidth, vh = video.videoHeight;
      const scale = Math.min(boxW / vw, boxH / vh);
      const cw = vw * scale, ch = vh * scale;
      const offX = (boxW - cw) / 2, offY = (boxH - ch) / 2;

      let current = null;
      liveTrace.forEach(detection => {
        if (Math.abs(Number(detection.time_sec) - video.currentTime) > 0.25) return;
        const [x0, y0, x1, y1] = detection.bbox || [];
        if (![x0, y0, x1, y1].every(Number.isFinite)) return;
        const color = ROI_HEX[safeRoi(detection.roi)];
        const x = offX + x0 * cw, y = offY + y0 * ch;
        const w = (x1 - x0) * cw, h = (y1 - y0) * ch;
        const lw = Math.max(1.5, boxW / 420);
        if (detection.flagged) {
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.18;
          ctx.fillRect(x, y, w, h);
          ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = detection.flagged ? color : 'rgba(255,255,255,0.55)';
        ctx.lineWidth = detection.flagged ? lw * 1.5 : lw;
        ctx.strokeRect(x, y, w, h);
        if (detection.flagged && !current) current = detection;
      });

      if (badge) {
        if (current) {
          badge.hidden = false;
          badge.textContent = `#${current.step} ${current.roi} \u00b7 ${formatDuration(Number(current.time_sec))} \u00b7 ${(clamp(Number(current.confidence), 0, 1) * 100).toFixed(0)}%`;
        } else {
          badge.hidden = true;
        }
      }
    };

    const redraw = () => { if (!video.paused) draw(); };
    video.addEventListener('loadedmetadata', draw);
    video.addEventListener('timeupdate', redraw);
    video.addEventListener('seeked', draw);
    video.addEventListener('play', draw);
    window.addEventListener('resize', draw);
    renderVideoCleanup = () => {
      video.removeEventListener('loadedmetadata', draw);
      video.removeEventListener('timeupdate', redraw);
      video.removeEventListener('seeked', draw);
      video.removeEventListener('play', draw);
      window.removeEventListener('resize', draw);
      renderVideoCleanup = () => {};
    };
  };

  const seekVideo = seconds => {
    const video = $('#liveVideo');
    if (!video) return;
    const t = clamp(Number(seconds) || 0, 0, Number.isFinite(video.duration) ? video.duration : 0);
    try { video.currentTime = t; } catch { /* not seekable yet */ }
  };

  /* -- Lightbox --------------------------------------------- */
  let lightboxIndex = 0;
  const setupLightbox = () => {
    $('#lightboxClose')?.addEventListener('click', closeLightbox);
    $('#lightbox')?.addEventListener('click', event => { if (event.target === $('#lightbox')) closeLightbox(); });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') { closeLightbox(); return; }
      if ($('#lightbox')?.hasAttribute('hidden')) return;
      if (event.key === 'ArrowLeft') stepLightbox(-1);
      if (event.key === 'ArrowRight') stepLightbox(1);
    });
    $('#lightboxPrev')?.addEventListener('click', () => stepLightbox(-1));
    $('#lightboxNext')?.addEventListener('click', () => stepLightbox(1));
  };

  const openLightbox = (index, trigger = null) => {
    const lightbox = $('#lightbox');
    if (!lightbox) return;
    lightboxTrigger = trigger || document.activeElement;
    const shown = visibleTrace();
    lightboxIndex = clamp(index, 0, Math.max(0, shown.length - 1));
    const detection = shown[lightboxIndex];
    const image = $('#lightboxImg');
    image.src = detection.image_url || '';
    image.alt = `Annotated frame ${detection.video_frame ?? ''}`;
    image.style.display = detection.image_url ? '' : 'none';
    setText('lightboxCap', `STEP #${detection.step} \u00b7 FRAME ${detection.video_frame ?? '\u2014'} \u00b7 ${safeRoi(detection.roi)} \u00b7 ${formatDuration(Number(detection.time_sec))} \u00b7 ${(Number(detection.confidence) * 100).toFixed(1)}% confidence${detection.flagged ? ' \u00b7 FLAGGED' : ''}`);
    $('#lightboxPrev').disabled = shown.length <= 1;
    $('#lightboxNext').disabled = shown.length <= 1;
    lightbox.removeAttribute('hidden');
    document.body.style.overflow = 'hidden';
    $('#lightboxClose')?.focus();
  };

  const stepLightbox = direction => {
    const shown = visibleTrace();
    if (shown.length <= 1) return;
    lightboxIndex = (lightboxIndex + direction + shown.length) % shown.length;
    openLightbox(lightboxIndex);
  };

  const closeLightbox = () => {
    $('#lightbox')?.setAttribute('hidden', '');
    document.body.style.overflow = '';
    lightboxTrigger?.focus?.();
    lightboxTrigger = null;
  };
})();
